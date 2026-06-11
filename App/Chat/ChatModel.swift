import Foundation
import OmniKit
import Observation

/// UI-facing state and orchestration for the "chat with your folder" feature.
///
/// Owns the optional ``ChatEngine`` (the local Qwen3 LLM), runs the RAG retrieval, and drives the
/// streaming transcript. All published state lives on the main actor; the heavy work (embedding the
/// question, vector search, token generation) runs off the main actor and streams results back.
@MainActor
@Observable
final class ChatModel {
    enum ModelState: Equatable {
        case notInstalled, downloading, installed, loading, ready, failed(String)
    }

    struct Turn: Identifiable {
        let id = UUID()
        let role: ChatMessage.Role
        var text: String
        var sources: [ChatSource] = []
        var isStreaming = false
        var stats: ChatGenerationStats? = nil
    }

    var modelState: ModelState = ChatModelLocator.isInstalled() ? .installed : .notInstalled
    var downloadFraction: Double = 0
    var downloadLabel: String = ""
    var isGenerating = false

    /// The folder this conversation is about (nil = all indexed files). Each scope keeps its own
    /// transcript, so switching folders switches conversations.
    private(set) var scope: URL?
    private var byScope: [String: [Turn]] = [:]
    private func key(_ folder: URL?) -> String { folder?.path ?? "" }

    /// The transcript for the current scope (read by the UI).
    var transcript: [Turn] { byScope[key(scope)] ?? [] }

    /// Point the chat at a folder (nil = all files). Its conversation is restored if one exists.
    func setScope(_ folder: URL?) { scope = folder }

    // Generation parameters, persisted across launches.
    var temperature: Double = UserDefaults.standard.object(forKey: "omni.chat.temperature") as? Double ?? 0.7 {
        didSet { UserDefaults.standard.set(temperature, forKey: "omni.chat.temperature") }
    }
    var topP: Double = UserDefaults.standard.object(forKey: "omni.chat.topP") as? Double ?? 0.8 {
        didSet { UserDefaults.standard.set(topP, forKey: "omni.chat.topP") }
    }
    var maxTokens: Int = UserDefaults.standard.object(forKey: "omni.chat.maxTokens") as? Int ?? 1024 {
        didSet { UserDefaults.standard.set(maxTokens, forKey: "omni.chat.maxTokens") }
    }

    /// Memory cap (GB) propagated from AppModel; the chat model needs headroom, so we refuse to load
    /// under a tight cap and unload if the user lowers it.
    var memoryCapGB: Double = 0

    private var omniEngine: OmniEngine?
    private var store: VectorStore?
    private var chatEngine: ChatEngine?
    private var genTask: Task<Void, Never>?

    /// Wire up the shared embedder + index. Called from AppModel.bootstrap, next to serving.attach.
    func attach(engine: OmniEngine, store: VectorStore) {
        self.omniEngine = engine
        self.store = store
    }

    // MARK: - Sending a message

    /// Ask a question about the current `scope` (nil = all indexed files). `chunkSetting` is the app's
    /// current maxCharsPerChunk, used to re-derive retrieved chunk text identically to indexing.
    func send(_ question: String, chunkSetting: Int) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !isGenerating, modelState != .downloading else { return }

        let sk = key(scope)            // pin this conversation to the scope it started in
        let folder = scope
        byScope[sk, default: []].append(Turn(role: .user, text: q))
        let assistant = Turn(role: .assistant, text: "", isStreaming: true)
        byScope[sk, default: []].append(assistant)
        let assistantId = assistant.id
        isGenerating = true

        let history = recentHistory(sk)
        let temp = Float(temperature), topp = Float(topP), maxTok = maxTokens

        genTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.ensureLoaded()
                guard let omni = self.omniEngine, let store = self.store, let chat = self.chatEngine else {
                    throw OmniError.model("chat is not ready")
                }
                var settings = IndexSettings.default
                settings.maxCharsPerChunk = chunkSetting

                // Embed + retrieve off the main actor.
                let ctx: ChatContext = await Task.detached {
                    let qv = omni.embedQuery(q)
                    return ChatContextBuilder.build(question: q, queryVector: qv, store: store,
                                                    folder: folder, settings: settings)
                }.value
                self.update(sk, assistantId) { $0.sources = ctx.sources }

                var messages = [ChatMessage(role: .system, content: ChatContext.systemPrompt)]
                messages.append(contentsOf: history)
                messages.append(ChatMessage(role: .user, content: ctx.userMessage))

                var params = ChatGenerationParams()
                params.temperature = temp; params.topP = topp; params.maxTokens = maxTok

                for try await chunk in chat.generate(messages: messages, params: params,
                                                     onStats: { [weak self] s in
                                                         Task { @MainActor in self?.update(sk, assistantId) { $0.stats = s } }
                                                     }) {
                    if Task.isCancelled { break }
                    self.update(sk, assistantId) { $0.text += chunk }
                }
                self.update(sk, assistantId) { $0.isStreaming = false }
            } catch {
                self.update(sk, assistantId) {
                    if $0.text.isEmpty { $0.text = "Sorry, I could not generate an answer: \(error)" }
                    $0.isStreaming = false
                }
            }
            self.isGenerating = false
        }
    }

    func stop() { genTask?.cancel() }

    func clearTranscript() {
        guard !isGenerating else { return }
        byScope[key(scope)] = []
    }

    // MARK: - Model lifecycle

    private func ensureLoaded() async throws {
        if memoryCapGB > 0 && memoryCapGB < 4 {
            modelState = .failed("Chat needs the memory limit at 4 GB or higher (Settings > Performance).")
            throw OmniError.model("memory cap too low for chat")
        }
        if chatEngine == nil {
            guard let dir = ChatModelLocator.resolve() else {
                modelState = .notInstalled
                throw OmniError.model("chat model not installed")
            }
            chatEngine = ChatEngine(modelDir: dir, gpuCoordinator: omniEngine)
        }
        if let chat = chatEngine, !chat.isLoaded {
            modelState = .loading
            try await chat.load()
            modelState = .ready
        } else if chatEngine?.isLoaded == true {
            modelState = .ready
        }
    }

    /// Unload the model when the memory cap drops below the chat threshold.
    func unloadIfOverCap(capGB: Double) {
        memoryCapGB = capGB
        if capGB > 0 && capGB < 4 {
            chatEngine?.unload()
            if modelState == .ready { modelState = ChatModelLocator.isInstalled() ? .installed : .notInstalled }
        }
    }

    // MARK: - Download / delete

    func downloadModel() {
        guard modelState != .downloading, let dest = ChatModelLocator.installDir() else { return }
        modelState = .downloading
        downloadFraction = 0
        downloadLabel = "Starting download..."
        Task { [weak self] in
            let downloader = ModelDownloader()
            do {
                try await downloader.download(repo: ChatModelLocator.repo, files: ChatModelLocator.files,
                                              to: dest) { [weak self] p in
                    Task { @MainActor in
                        guard let self else { return }
                        let frac = p.total > 0 ? Double(p.received) / Double(p.total) : 0
                        self.downloadFraction = (Double(p.fileIndex) + frac) / Double(max(1, p.fileCount))
                        self.downloadLabel = "Downloading \(p.file)..."
                    }
                }
                await MainActor.run { self?.modelState = .installed; self?.downloadFraction = 1 }
            } catch {
                await MainActor.run { self?.modelState = .failed("Download failed: \(error)") }
            }
        }
    }

    func deleteModel() {
        chatEngine?.unload()
        chatEngine = nil
        try? ChatModelLocator.delete()
        modelState = .notInstalled
        byScope.removeAll()
    }

    // MARK: - Helpers

    private func recentHistory(_ sk: String, limit: Int = 6) -> [ChatMessage] {
        // Completed turns only (exclude the just-appended user question + streaming placeholder).
        let done = (byScope[sk] ?? []).dropLast(2)
        return done.suffix(limit).map { t in
            ChatMessage(role: t.role,
                        content: t.role == .assistant ? ChatTemplate.cleanAssistantContent(t.text) : t.text)
        }
    }

    private func update(_ sk: String, _ id: UUID, _ mutate: (inout Turn) -> Void) {
        guard var turns = byScope[sk], let i = turns.firstIndex(where: { $0.id == id }) else { return }
        mutate(&turns[i])
        byScope[sk] = turns
    }
}
