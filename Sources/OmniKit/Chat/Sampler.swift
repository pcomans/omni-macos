import Foundation

/// A tiny, fully deterministic pseudo-random generator (SplitMix64).
///
/// LEARNING NOTE - why a custom RNG
/// -------------------------------
/// Sampling needs randomness, but for tests and reproducibility we want to be able to fix a seed and
/// get the exact same sequence every run. SplitMix64 is a well-known 64-bit generator that is one
/// line of arithmetic per draw and has good statistical quality - ideal for "give me repeatable
/// randomness with no dependencies."
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Picks the next token id from a row of logits using temperature + nucleus (top-p) sampling.
///
/// LEARNING NOTE - logits, temperature, softmax
/// --------------------------------------------
/// The model outputs a "logit" (an unnormalized score) for every one of the ~152k vocabulary tokens.
/// Softmax turns logits into a probability distribution: `p_i = exp(logit_i) / sum_j exp(logit_j)`.
/// "Temperature" rescales the logits before softmax (`logit_i / T`): T < 1 sharpens the distribution
/// (more confident, more repetitive), T > 1 flattens it (more random), and T = 0 means "always take
/// the single highest logit" (greedy / argmax), which is what the verification fixtures use because
/// it is deterministic.
///
/// LEARNING NOTE - nucleus (top-p) sampling
/// ----------------------------------------
/// Sampling from the full distribution occasionally picks a very unlikely, incoherent token. Top-p
/// fixes this: sort tokens by probability, keep the smallest set whose probabilities sum to at least
/// `p` (the "nucleus"), zero out the rest, renormalize, and sample from what remains. So the model
/// can still be creative among plausible continuations but almost never derails into nonsense.
///
/// LEARNING NOTE - why we only look at the top 512 logits
/// -----------------------------------------------------
/// A 152k-way softmax on the CPU every token would be wasteful, and the probability mass outside the
/// few hundred most likely tokens is negligible for any sane temperature/top-p. So we first select
/// the 512 highest logits, then do softmax/top-p on just those. This keeps per-token CPU cost well
/// under a millisecond. (Sampling on the CPU, not the GPU, also keeps the RNG deterministic and off
/// the shared GPU stream the embedder uses.)
struct Sampler {
    var temperature: Float
    var topP: Float
    private var rng: SplitMix64

    init(temperature: Float, topP: Float, seed: UInt64) {
        self.temperature = temperature
        self.topP = topP
        self.rng = SplitMix64(seed: seed)
    }

    /// Choose a token id from `logits` (length == vocab size, fp32).
    mutating func sample(_ logits: [Float]) -> Int {
        if temperature <= 0 { return argmax(logits) }

        // 1. Select the top `k` logit indices (k small relative to vocab).
        let k = min(512, logits.count)
        let topIdx = topKIndices(logits, k: k)

        // 2. Softmax over just those, with temperature, numerically stabilized by max-subtraction.
        let invT = 1 / temperature
        var maxLogit = -Float.greatestFiniteMagnitude
        for i in topIdx { maxLogit = max(maxLogit, logits[i]) }
        var probs = [Float](repeating: 0, count: topIdx.count)
        var sum: Float = 0
        for (j, i) in topIdx.enumerated() {
            let e = expf((logits[i] - maxLogit) * invT)
            probs[j] = e
            sum += e
        }
        for j in probs.indices { probs[j] /= sum }

        // 3. Nucleus cut: walk the already-descending list, keep until cumulative >= topP.
        //    topKIndices returns indices sorted by descending logit, so probs is descending too.
        var cutoff = topIdx.count
        if topP < 1 {
            var cum: Float = 0
            for j in probs.indices {
                cum += probs[j]
                if cum >= topP { cutoff = j + 1; break }
            }
        }

        // 4. Renormalize the kept nucleus and draw.
        var keptSum: Float = 0
        for j in 0 ..< cutoff { keptSum += probs[j] }
        let r = Float(rng.next() >> 11) / Float(1 << 53) * keptSum   // uniform in [0, keptSum)
        var acc: Float = 0
        for j in 0 ..< cutoff {
            acc += probs[j]
            if r < acc { return topIdx[j] }
        }
        return topIdx[cutoff - 1]
    }

    private func argmax(_ logits: [Float]) -> Int {
        var best = 0
        var bestV = logits[0]
        for i in 1 ..< logits.count where logits[i] > bestV { bestV = logits[i]; best = i }
        return best
    }

    /// Indices of the `k` largest logits, sorted by descending logit. Partial selection via a
    /// fixed-size min-heap keyed on logit value, so cost is O(n log k), not a full O(n log n) sort.
    private func topKIndices(_ logits: [Float], k: Int) -> [Int] {
        var heap = [Int]()   // holds candidate indices; heap[0] is the smallest logit kept so far
        heap.reserveCapacity(k)
        func siftDown(_ start: Int) {
            var root = start
            while true {
                let l = 2 * root + 1, r = 2 * root + 2
                var smallest = root
                if l < heap.count && logits[heap[l]] < logits[heap[smallest]] { smallest = l }
                if r < heap.count && logits[heap[r]] < logits[heap[smallest]] { smallest = r }
                if smallest == root { break }
                heap.swapAt(root, smallest); root = smallest
            }
        }
        func siftUp(_ start: Int) {
            var child = start
            while child > 0 {
                let parent = (child - 1) / 2
                if logits[heap[child]] < logits[heap[parent]] { heap.swapAt(child, parent); child = parent }
                else { break }
            }
        }
        for i in logits.indices {
            if heap.count < k {
                heap.append(i); siftUp(heap.count - 1)
            } else if logits[i] > logits[heap[0]] {
                heap[0] = i; siftDown(0)
            }
        }
        return heap.sorted { logits[$0] > logits[$1] }
    }
}
