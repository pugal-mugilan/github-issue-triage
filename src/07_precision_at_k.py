"""
07_precision_at_k.py — Precision@K evaluation
==============================================
Computes precision@5 and precision@10 for the repo-weighted NN,
both overall and per-repo.

This is the formal success criterion from PROBLEM.md:
  - In-domain P@5 ≥ 0.70

Why P@K matters:
  The maintainer sees only the top K issues the model ranks highest.
  P@K = "of those top K, how many actually resolved within 168h?"
  A model with decent F1 but bad P@K puts wrong issues at the top.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"


# ── Network architecture (must match 06b_nn_weighted.py) ─────────────
class IssueTriage_NN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def precision_at_k(y_true, probs, k):
    """
    Sort by predicted probability (highest first),
    check how many of the top K are actually positive.
    """
    if len(y_true) < k:
        return None
    top_k_indices = np.argsort(probs)[::-1][:k]
    top_k_labels  = y_true[top_k_indices]
    return top_k_labels.sum() / k


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # Load test data
    X_test = pd.read_parquet(PROCESSED / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

    print(f"Test set: {len(X_test):,} rows, {y_test.sum():,} positives "
          f"({y_test.mean():.1%} positive rate)\n")

    # Load trained weights
    n_features = X_test.shape[1]
    model = IssueTriage_NN(n_features)
    state_dict = torch.load(MODELS_DIR / "nn_weighted_model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Score all test issues
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_te_t)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    y_true = y_test.values

    # Overall P@K
    print("=" * 60)
    print("OVERALL Precision@K (all repos mixed)")
    print("=" * 60)

    overall_results = {}
    for k in [5, 10, 20]:
        p_at_k = precision_at_k(y_true, probs, k)
        overall_results[f"p_at_{k}"] = round(float(p_at_k), 4) if p_at_k is not None else None
        print(f"  P@{k:<3d} = {p_at_k:.4f}  ({int(p_at_k * k)}/{k} correct)")

    # Per-repo P@K
    REPO_COLS = [c for c in X_test.columns if c.startswith("repo_")]

    print(f"\n{'=' * 60}")
    print("PER-REPO Precision@K")
    print("=" * 60)

    per_repo_results = {}
    for col in REPO_COLS:
        repo_name = col.replace("repo_", "")
        mask = X_test[col].values == 1

        repo_y    = y_true[mask]
        repo_prob = probs[mask]
        n_total   = mask.sum()
        n_pos     = repo_y.sum()

        print(f"\n  {repo_name}")
        print(f"  {'─' * 50}")
        print(f"  Test rows: {n_total:,} | Positives: {int(n_pos):,} "
              f"({n_pos / n_total:.1%})")

        repo_pk = {}
        for k in [5, 10]:
            if n_total < k:
                print(f"  P@{k:<3d} = N/A (only {n_total} rows in test)")
                continue
            p = precision_at_k(repo_y, repo_prob, k)
            repo_pk[f"p_at_{k}"] = round(float(p), 4)
            print(f"  P@{k:<3d} = {p:.4f}  ({int(p * k)}/{k} correct)")
        per_repo_results[repo_name] = repo_pk

    # Success criteria check
    overall_p5 = precision_at_k(y_true, probs, 5)

    print(f"\n{'=' * 60}")
    print("SUCCESS CRITERIA CHECK (from PROBLEM.md)")
    print("=" * 60)
    print(f"  In-domain P@5:  {overall_p5:.4f}  "
          f"{'✅ PASS' if overall_p5 >= 0.70 else '❌ FAIL'}  (threshold: ≥ 0.70)")

    # Probability distribution diagnostic
    print(f"\n{'=' * 60}")
    print("PROBABILITY DISTRIBUTION DIAGNOSTIC")
    print("=" * 60)
    print(f"  Min:    {probs.min():.4f}")
    print(f"  25th:   {np.percentile(probs, 25):.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  75th:   {np.percentile(probs, 75):.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Std:    {probs.std():.4f}")

    print(f"\n  Top 10 probabilities:")
    top10_idx = np.argsort(probs)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        actual = "✅ YES" if y_true[idx] == 1 else "❌ NO"
        print(f"    #{rank:2d}  prob={probs[idx]:.4f}  actual={actual}")

    # Return metrics for run_pipeline.py
    metrics = {
        "in_domain_p_at_5": round(float(overall_p5), 4),
        "in_domain_p_at_5_pass": bool(overall_p5 >= 0.70),
        "overall": overall_results,
        "per_repo": per_repo_results,
        "prob_distribution": {
            "min": round(float(probs.min()), 4),
            "median": round(float(np.median(probs)), 4),
            "max": round(float(probs.max()), 4),
            "std": round(float(probs.std()), 4),
        },
    }
    return metrics


if __name__ == "__main__":
    main()