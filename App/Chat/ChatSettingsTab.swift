import SwiftUI
import OmniKit

/// Settings > Chat: download/remove the optional local chat model and tune generation.
struct ChatTab: View {
    @Environment(AppModel.self) private var model: AppModel
    private var chat: ChatModel { model.chat }

    var body: some View {
        Form {
            Section {
                modelRow
            } header: {
                Text("Local chat model")
            } footer: {
                Text("Chat answers questions about your indexed files using Qwen3-1.7B running on your Mac (about 1 GB download, ~1 GB memory while loaded, plus up to ~0.5 GB while answering). It is optional; search works without it. Requires the memory limit at 4 GB or higher.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Creativity (temperature)")
                        Spacer()
                        Text(String(format: "%.2f", chat.temperature)).foregroundStyle(.secondary).monospacedDigit()
                    }
                    Slider(value: Binding(get: { chat.temperature }, set: { chat.temperature = $0 }), in: 0 ... 1.5)
                        .labelsHidden()
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Nucleus (top-p)")
                        Spacer()
                        Text(String(format: "%.2f", chat.topP)).foregroundStyle(.secondary).monospacedDigit()
                    }
                    Slider(value: Binding(get: { chat.topP }, set: { chat.topP = $0 }), in: 0.1 ... 1.0)
                        .labelsHidden()
                }
                Picker("Max answer length", selection: Binding(get: { chat.maxTokens }, set: { chat.maxTokens = $0 })) {
                    Text("Short \u{00B7} 256").tag(256)
                    Text("Medium \u{00B7} 512").tag(512)
                    Text("Long \u{00B7} 1024").tag(1024)
                    Text("Very long \u{00B7} 2048").tag(2048)
                }
            } header: {
                Text("Generation")
            } footer: {
                Text("Lower temperature gives more focused, repeatable answers; higher is more varied.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder private var modelRow: some View {
        switch chat.modelState {
        case .notInstalled:
            HStack {
                Label("Not installed", systemImage: "circle.dashed").foregroundStyle(.secondary)
                Spacer()
                Button("Download") { chat.downloadModel() }.buttonStyle(.borderedProminent)
            }
        case .downloading:
            VStack(alignment: .leading, spacing: 6) {
                Text(chat.downloadLabel).font(.callout)
                ProgressView(value: chat.downloadFraction)
            }
        case .installed, .loading, .ready:
            HStack {
                Label(chat.modelState == .ready ? "Installed (loaded)" : "Installed",
                      systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                Spacer()
                Button("Remove") { chat.deleteModel() }
            }
        case .failed(let msg):
            VStack(alignment: .leading, spacing: 6) {
                Label("Problem", systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(msg).font(.caption).foregroundStyle(.secondary)
                Button("Try download again") { chat.downloadModel() }
            }
        }
    }
}
