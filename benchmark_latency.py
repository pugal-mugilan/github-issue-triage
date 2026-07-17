"""
Latency benchmark for github-issue-triage API.
Sends 50 synthetic issues to /predict/batch, measures p50/p95/p99.
Success criterion: total batch latency ≤ 500ms (PROBLEM.md).
"""

import time
import requests
import statistics

BASE_URL = "http://localhost:8000"

# --- 50 synthetic issues covering in-domain, weak-slice, and OOD repos ---

REPOS = [
    "pytorch/pytorch", "pytorch/pytorch", "pytorch/pytorch",
    "langchain-ai/langchain", "langchain-ai/langchain",
    "fastapi/fastapi", "numpy/numpy", "pandas-dev/pandas",
    "scikit-learn/scikit-learn", "huggingface/transformers",
]

TITLES = [
    "Bug: model crashes on empty input",
    "Feature request: add batch processing support",
    "TypeError in forward pass with mixed dtypes",
    "Documentation typo in README quickstart",
    "Performance regression after upgrade to v2.0",
    "Add support for custom loss functions",
    "Memory leak when training with large datasets",
    "Fix: incorrect gradient computation for conv layers",
    "Question: how to fine-tune on custom dataset?",
    "CI failing on Python 3.12 with segfault",
]

BODIES = [
    "I'm getting a crash when I pass an empty tensor.\n```python\nmodel(torch.tensor([]))\n```\nTraceback:\n```\nRuntimeError: empty input\n```",
    "It would be great to support batch inference natively. Currently I loop one-by-one which is slow.",
    "When mixing float16 and float32 tensors, the forward pass throws TypeError. Steps to reproduce attached.",
    "Small typo on line 42 of README.md — 'teh' should be 'the'.",
    "After upgrading, training time increased 3x. Profiler shows bottleneck in data loader. See attached flamegraph.",
    "Would love to pass a custom loss function to the trainer. Currently only CrossEntropy is supported.",
    "RAM usage grows linearly during training and never drops. Looks like tensors aren't being freed. Using v1.13.",
    "The gradient for depthwise conv is wrong when groups > 1. Numerical gradient check fails. Minimal repro attached.",
    "I have a custom NER dataset. What's the recommended way to fine-tune? The docs mention Trainer but no full example.",
    "CI segfaults on Python 3.12 only. Works fine on 3.11. Bisected to commit abc123. Logs attached.",
]

ASSOCIATIONS = ["MEMBER", "CONTRIBUTOR", "NONE", "COLLABORATOR", "NONE"]


def build_issues(n=50):
    """Generate n synthetic issues by cycling through templates."""
    issues = []
    for i in range(n):
        issues.append({
            "title": TITLES[i % len(TITLES)],
            "body": BODIES[i % len(BODIES)],
            "repo": REPOS[i % len(REPOS)],
            "author_association": ASSOCIATIONS[i % len(ASSOCIATIONS)],
            "user_login": f"user_{i}",
            "created_at": "2025-06-15T10:00:00Z",
        })
    return issues


def benchmark_batch(issues):
    """Send all issues to /predict/batch, return elapsed time in ms."""
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/predict/batch", json={"issues": issues})
    elapsed_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return elapsed_ms, resp.json()


def benchmark_individual(issues):
    """Send each issue to /predict individually, return list of times in ms."""
    times = []
    for issue in issues:
        start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/predict", json=issue)
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        times.append(elapsed_ms)
    return times


def main():
    issues = build_issues(50)

    # --- Warmup: one request to avoid cold-start bias ---
    requests.post(f"{BASE_URL}/predict", json=issues[0])

    # --- Batch benchmark (the PROBLEM.md criterion) ---
    print("=" * 60)
    print("BATCH BENCHMARK: 50 issues → /predict/batch")
    print("=" * 60)

    batch_times = []
    for run in range(5):
        elapsed, result = benchmark_batch(issues)
        batch_times.append(elapsed)
        print(f"  Run {run + 1}: {elapsed:.1f} ms  ({len(result['predictions'])} predictions)")

    print(f"\n  Batch p50:  {statistics.median(batch_times):.1f} ms")
    print(f"  Batch mean: {statistics.mean(batch_times):.1f} ms")
    print(f"  Batch min:  {min(batch_times):.1f} ms")
    print(f"  Batch max:  {max(batch_times):.1f} ms")

    target = 500
    passed = statistics.median(batch_times) <= target
    print(f"\n  PROBLEM.md target: ≤ {target} ms")
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")

    # --- Individual benchmark (for p50/p95/p99 per-request) ---
    print("\n" + "=" * 60)
    print("INDIVIDUAL BENCHMARK: 50 issues → /predict (one at a time)")
    print("=" * 60)

    times = benchmark_individual(issues)
    times_sorted = sorted(times)

    print(f"\n  p50:  {times_sorted[24]:.1f} ms")
    print(f"  p95:  {times_sorted[47]:.1f} ms")
    print(f"  p99:  {times_sorted[49]:.1f} ms")  # last item for n=50
    print(f"  mean: {statistics.mean(times):.1f} ms")
    print(f"  min:  {min(times):.1f} ms")
    print(f"  max:  {max(times):.1f} ms")


if __name__ == "__main__":
    main()