import Foundation

/// Finds (and describes where to install) the local Qwen3-1.7B chat weights.
///
/// Mirrors the embedder's `ModelLocator`: the chat model is an OPTIONAL, separate download, so the
/// app must run perfectly with it absent. Resolution order:
///   1. `OMNI_CHAT_MODEL_DIR` env override (used by tests and the fixture/verify tooling),
///   2. the app's install directory (~/Library/Application Support/Omni/chat-qwen3-1.7b-4bit),
///   3. the Hugging Face hub cache, if the user already pulled it via huggingface_hub.
public enum ChatModelLocator {
    public static let repo = "Qwen/Qwen3-1.7B-MLX-4bit"
    public static let installFolderName = "chat-qwen3-1.7b-4bit"

    /// Files the Swift runtime needs at inference time (no Python, no GGUF).
    public static let files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]

    /// Where a downloaded chat model is installed.
    public static func installDir() -> URL? {
        let fm = FileManager.default
        guard let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                       appropriateFor: nil, create: true) else { return nil }
        return appSup.appendingPathComponent("Omni/\(installFolderName)")
    }

    /// True when a complete model (safetensors + config + tokenizer) is present at `dir`.
    static func isComplete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        for f in ["model.safetensors", "config.json", "tokenizer.json"] {
            if !fm.fileExists(atPath: dir.appendingPathComponent(f).path) { return false }
        }
        return true
    }

    /// First complete model directory in priority order, or nil if the chat model is not installed.
    public static func resolve() -> URL? {
        if let env = ProcessInfo.processInfo.environment["OMNI_CHAT_MODEL_DIR"] {
            let dir = URL(fileURLWithPath: env)
            if isComplete(dir) { return dir }
        }
        if let install = installDir(), isComplete(install) { return install }
        if let hub = hubSnapshot(), isComplete(hub) { return hub }
        return nil
    }

    public static func isInstalled() -> Bool { resolve() != nil }

    /// Remove the installed chat model (frees ~1 GB). Hub-cached copies are left untouched.
    public static func delete() throws {
        guard let install = installDir() else { return }
        if FileManager.default.fileExists(atPath: install.path) {
            try FileManager.default.removeItem(at: install)
        }
    }

    /// Newest snapshot directory under the HF hub cache for `repo`, if any.
    private static func hubSnapshot() -> URL? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let repoDir = home
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        guard let snaps = try? fm.contentsOfDirectory(at: repoDir, includingPropertiesForKeys: nil) else { return nil }
        return snaps.first { isComplete($0) }
    }
}
