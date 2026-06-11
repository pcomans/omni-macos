import XCTest
@testable import OmniKit

/// Tests for the token sampler. No model needed: it operates on plain logit arrays.
final class SamplerTests: XCTestCase {
    func testGreedyIsArgmax() {
        var s = Sampler(temperature: 0, topP: 1, seed: 1)
        let logits: [Float] = [0.1, 5.0, -2.0, 4.9, 0.0]
        XCTAssertEqual(s.sample(logits), 1)
    }

    func testSeededDeterminism() {
        let logits = (0 ..< 100).map { Float($0 % 7) }   // some structure, several ties
        var a = Sampler(temperature: 1.0, topP: 0.95, seed: 12345)
        var b = Sampler(temperature: 1.0, topP: 0.95, seed: 12345)
        let seqA = (0 ..< 50).map { _ in a.sample(logits) }
        let seqB = (0 ..< 50).map { _ in b.sample(logits) }
        XCTAssertEqual(seqA, seqB, "same seed must yield the same sample sequence")
    }

    func testDifferentSeedsDiffer() {
        let logits = (0 ..< 100).map { Float(($0 * 13) % 11) }
        var a = Sampler(temperature: 1.0, topP: 0.99, seed: 1)
        var b = Sampler(temperature: 1.0, topP: 0.99, seed: 2)
        let seqA = (0 ..< 80).map { _ in a.sample(logits) }
        let seqB = (0 ..< 80).map { _ in b.sample(logits) }
        XCTAssertNotEqual(seqA, seqB)
    }

    func testNucleusStaysInTopSet() {
        // Two tokens dominate; with a tight top-p only those two should ever be drawn.
        var logits = [Float](repeating: -20, count: 200)
        logits[10] = 5.0
        logits[20] = 4.8
        var s = Sampler(temperature: 1.0, topP: 0.9, seed: 7)
        let allowed: Set<Int> = [10, 20]
        for _ in 0 ..< 500 {
            XCTAssertTrue(allowed.contains(s.sample(logits)), "nucleus sampling drew outside the top set")
        }
    }
}
