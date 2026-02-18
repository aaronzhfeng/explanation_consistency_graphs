#!/usr/bin/env python
"""
Smoke test for API-based explanation generation.
Tests single, batch, and high-concurrency generation via OpenRouter.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecg.explain_llm import APIExplanationGenerator, generate_batch_with_stability

API_KEY_FILE = str(Path(__file__).parent.parent / "openroutera_api.txt")


def test_single():
    """Test single explanation generation."""
    print("=" * 60)
    print("Test 1: Single explanation")
    print("=" * 60)

    gen = APIExplanationGenerator.from_key_file(
        API_KEY_FILE,
        model_name="qwen/qwen3-8b",
        max_concurrency=10,
    )

    sentence = "This movie was absolutely terrible, a complete waste of time."
    start = time.time()
    exp = gen.generate_single(sentence, temperature=0.0)
    elapsed = time.time() - start

    print(f"  Sentence: {sentence}")
    print(f"  pred_label: {exp.pred_label}")
    print(f"  evidence: {exp.evidence}")
    print(f"  rationale: {exp.rationale}")
    print(f"  counterfactual: {exp.counterfactual}")
    print(f"  confidence: {exp.confidence}")
    print(f"  parse_success: {exp.parse_success}")
    print(f"  Time: {elapsed:.2f}s")
    assert exp.parse_success, f"Parse failed! Raw: {exp.raw_output}"
    print("  PASSED")


def test_batch_small():
    """Test small batch (10 examples)."""
    print("\n" + "=" * 60)
    print("Test 2: Small batch (10 examples)")
    print("=" * 60)

    gen = APIExplanationGenerator.from_key_file(
        API_KEY_FILE,
        model_name="qwen/qwen3-8b",
        max_concurrency=10,
    )

    sentences = [
        "This movie was absolutely terrible, a complete waste of time.",
        "I loved every minute of this film, truly a masterpiece!",
        "The acting was mediocre at best, but the story was engaging.",
        "A boring, predictable plot with no redeeming qualities.",
        "Surprisingly good! The twist at the end caught me off guard.",
        "Not my cup of tea, but I can see why others might enjoy it.",
        "An instant classic that will be remembered for decades.",
        "The worst film I've seen this year, hands down.",
        "A delightful comedy that had the whole audience laughing.",
        "Completely overrated, I don't understand the hype at all.",
    ]

    start = time.time()
    results = gen.generate_batch(sentences, temperature=0.0, show_progress=True)
    elapsed = time.time() - start

    success_count = sum(1 for r in results if r.parse_success)
    print(f"\n  Results: {success_count}/{len(results)} parsed successfully")
    print(f"  Time: {elapsed:.2f}s ({elapsed/len(sentences):.2f}s/example)")

    for i, (sent, exp) in enumerate(zip(sentences, results)):
        status = "OK" if exp.parse_success else "FAIL"
        print(f"  [{status}] {sent[:50]}... -> {exp.pred_label} (conf={exp.confidence})")

    gen.print_stats()
    assert success_count >= 8, f"Too many failures: {len(results) - success_count}"
    print("  PASSED")


def test_batch_concurrent():
    """Test higher concurrency (50 examples, 30 concurrent)."""
    print("\n" + "=" * 60)
    print("Test 3: Concurrency stress test (50 examples, 30 concurrent)")
    print("=" * 60)

    gen = APIExplanationGenerator.from_key_file(
        API_KEY_FILE,
        model_name="qwen/qwen3-8b",
        max_concurrency=30,
    )

    # Generate 50 diverse sentences
    templates = [
        "This movie was {}.",
        "I {} this restaurant experience.",
        "The book was {}, I couldn't put it down.",
        "What a {} performance by the lead actor.",
        "The service was {}, we waited forever.",
    ]
    adjectives_pos = ["amazing", "wonderful", "fantastic", "brilliant", "outstanding",
                      "superb", "excellent", "delightful", "marvelous", "spectacular"]
    adjectives_neg = ["terrible", "awful", "horrible", "dreadful", "disappointing",
                      "mediocre", "boring", "forgettable", "atrocious", "unbearable"]

    sentences = []
    for i in range(50):
        template = templates[i % len(templates)]
        if i % 2 == 0:
            adj = adjectives_pos[i % len(adjectives_pos)]
        else:
            adj = adjectives_neg[i % len(adjectives_neg)]
        sentences.append(template.format(adj))

    start = time.time()
    results = gen.generate_batch(sentences, temperature=0.0, show_progress=True)
    elapsed = time.time() - start

    success_count = sum(1 for r in results if r.parse_success)
    print(f"\n  Results: {success_count}/{len(results)} parsed successfully")
    print(f"  Time: {elapsed:.2f}s ({elapsed/len(sentences):.2f}s/example)")
    print(f"  Throughput: {len(sentences)/elapsed:.1f} examples/sec")

    gen.print_stats()
    assert success_count >= 40, f"Too many failures: {len(results) - success_count}"
    print("  PASSED")


def test_stability_sampling():
    """Test stability sampling with API backend."""
    print("\n" + "=" * 60)
    print("Test 4: Stability sampling (5 examples, 3 samples each)")
    print("=" * 60)

    gen = APIExplanationGenerator.from_key_file(
        API_KEY_FILE,
        model_name="qwen/qwen3-8b",
        max_concurrency=15,
    )

    sentences = [
        "This movie was absolutely terrible, a complete waste of time.",
        "I loved every minute of this film, truly a masterpiece!",
        "The acting was mediocre at best, but the story was engaging.",
        "A boring, predictable plot with no redeeming qualities.",
        "Surprisingly good! The twist at the end caught me off guard.",
    ]

    start = time.time()
    results = generate_batch_with_stability(
        gen, sentences, n_samples=3, sample_temperature=0.7, show_progress=True,
    )
    elapsed = time.time() - start

    for i, r in enumerate(results):
        print(f"\n  Example {i}: {sentences[i][:50]}...")
        print(f"    Primary: {r.primary.pred_label} (conf={r.primary.confidence})")
        print(f"    Reliability: {r.stability.reliability_score:.3f}")
        print(f"    Label agreement: {r.stability.label_agreement:.3f}")
        print(f"    Evidence Jaccard: {r.stability.evidence_jaccard:.3f}")
        print(f"    Rationale sim: {r.stability.rationale_similarity:.3f}")

    print(f"\n  Time: {elapsed:.2f}s")
    gen.print_stats()
    print("  PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["single", "batch", "concurrent", "stability", "all"],
                        default="all")
    args = parser.parse_args()

    if args.test in ("single", "all"):
        test_single()
    if args.test in ("batch", "all"):
        test_batch_small()
    if args.test in ("concurrent", "all"):
        test_batch_concurrent()
    if args.test in ("stability", "all"):
        test_stability_sampling()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
