import SwiftUI
import OmniKit

/// The chat pane shown in the detail column when the view mode is `.chat`. Scope follows the app's
/// folder filter; answers cite the files they used as clickable chips.
struct ChatPane: View {
    @Environment(AppModel.self) private var model: AppModel
    @Environment(\.openSettings) private var openSettings
    @State private var draft = ""

    private var chat: ChatModel { model.chat }
    private var scopeName: String { chat.scope?.lastPathComponent ?? "all indexed files" }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .quickLookPreview(Binding(get: { model.previewURL }, set: { model.previewURL = $0 }))
        // The folder filter is the single source of truth for the chat scope; keep the conversation
        // in sync with it however chat was entered (header button, sidebar, or the keyboard shortcut).
        .onAppear { chat.setScope(model.filterFolder) }
        .onChange(of: model.filterFolder) { _, f in chat.setScope(f) }
    }

    @ViewBuilder private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "text.bubble").foregroundStyle(.tint)
            Text("Chatting with \(scopeName)").font(.callout).fontWeight(.medium)
            Spacer()
            if !chat.transcript.isEmpty {
                Button { chat.clearTranscript() } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless).disabled(chat.isGenerating)
                    .help("Clear conversation")
            }
            Button { model.exitChat() } label: { Label("Done", systemImage: "xmark.circle.fill") }
                .buttonStyle(.borderless)
                .help("Back to the folder view")
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
    }

    @ViewBuilder private var content: some View {
        switch chat.modelState {
        case .notInstalled, .downloading, .failed:
            modelGate
        default:
            if model.indexedFileCount == 0 {
                CenteredStatus(symbol: "tray", title: "Nothing indexed yet",
                               subtitle: "Add and index a folder to chat about its contents.", showSpinner: false)
            } else {
                conversation
            }
        }
    }

    @ViewBuilder private var modelGate: some View {
        switch chat.modelState {
        case .downloading:
            VStack(spacing: 12) {
                ProgressView(value: chat.downloadFraction)
                    .progressViewStyle(.linear).frame(maxWidth: 320)
                Text(chat.downloadLabel).font(.callout).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
        case .failed(let msg):
            CenteredStatus(symbol: "exclamationmark.triangle", title: "Chat unavailable",
                           subtitle: msg, showSpinner: false,
                           action: ("Open Settings", { openSettings() }))
        default:
            CenteredStatus(symbol: "text.bubble", title: "Download the chat model",
                           subtitle: "Chat runs a local model (about 1 GB) to answer questions about your files. It stays on your Mac.",
                           showSpinner: false,
                           action: ("Get the chat model", { openSettings() }))
        }
    }

    @ViewBuilder private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    if chat.transcript.isEmpty {
                        Text("Ask a question about \(scopeName).")
                            .font(.callout).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center).padding(.top, 40)
                    }
                    ForEach(chat.transcript) { turn in
                        TurnView(turn: turn, model: model)
                            .id(turn.id)
                    }
                }
                .padding(14)
            }
            .onChange(of: chat.transcript.last?.text) {
                if let last = chat.transcript.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
        inputBar
    }

    @ViewBuilder private var inputBar: some View {
        Divider()
        HStack(spacing: 8) {
            TextField("Ask about these files...", text: $draft, axis: .vertical)
                .textFieldStyle(.plain).lineLimit(1 ... 5)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .onSubmit(send)
            if chat.isGenerating {
                Button { chat.stop() } label: { Image(systemName: "stop.fill") }
                    .help("Stop")
            } else {
                Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title2) }
                    .buttonStyle(.borderless)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(10)
    }

    private func send() {
        let q = draft
        guard !q.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !chat.isGenerating else { return }
        draft = ""
        chat.send(q, chunkSetting: model.maxTextChunkChars)
    }
}

/// One transcript turn: user bubble (trailing) or assistant text with citation chips (leading).
private struct TurnView: View {
    let turn: ChatModel.Turn
    let model: AppModel

    var body: some View {
        if turn.role == .user {
            HStack {
                Spacer(minLength: 40)
                Text(turn.text)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(RoundedRectangle(cornerRadius: 12).fill(.quaternary))
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                if turn.text.isEmpty && turn.isStreaming {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(turn.sources.isEmpty ? "Searching your files\u{2026}" : "Thinking\u{2026}")
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text(turn.text)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if !turn.sources.isEmpty {
                    citations
                }
            }
        }
    }

    @ViewBuilder private var citations: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sources").font(.caption).foregroundStyle(.tertiary)
            FlowLayout(spacing: 6) {
                ForEach(turn.sources) { src in
                    Button {
                        NSWorkspace.shared.openAsync(URL(fileURLWithPath: src.path))
                    } label: {
                        let name = URL(fileURLWithPath: src.path).lastPathComponent
                        Text(src.locator.isEmpty ? "[\(src.id)] \(name)" : "[\(src.id)] \(name) \u{00b7} \(src.locator)")
                            .font(.caption).lineLimit(1)
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(Capsule().fill(.quaternary))
                    }
                    .buttonStyle(.plain)
                    .help(src.path)
                    .contextMenu {
                        Button("Open") { NSWorkspace.shared.openAsync(URL(fileURLWithPath: src.path)) }
                        Button("Reveal in Finder") { NSWorkspace.shared.revealAsync(URL(fileURLWithPath: src.path)) }
                        Button("Quick Look") { model.previewURL = URL(fileURLWithPath: src.path) }
                    }
                }
            }
        }
    }
}

/// A simple wrapping HStack for citation chips (no fixed column count).
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > maxWidth && x > 0 { x = 0; y += rowHeight + spacing; rowHeight = 0 }
            x += s.width + spacing; rowHeight = max(rowHeight, s.height)
        }
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = bounds.minX, y: CGFloat = bounds.minY, rowHeight: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > bounds.minX + maxWidth && x > bounds.minX { x = bounds.minX; y += rowHeight + spacing; rowHeight = 0 }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing; rowHeight = max(rowHeight, s.height)
        }
    }
}
