"""
generate_golden_set.py — Create frozen golden test set for regression testing.
================================================================================
Run ONCE after confirming model is working correctly. Produces:

  tests/golden/golden_set.json — 25 issues with raw fields + expected predictions

This file is the fixed reference point. Regression tests compare future model
runs against these frozen expectations.

Usage:
  python generate_golden_set.py

Requires: trained model artifacts in models/, processed data in data/processed/
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


# ── Model architecture (must match training) ────────────────────────
class _IssueTriage_NN(nn.Module):
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

def load_model(encoders_path, model_path):
    """Load encoders and model weights, return (model, feature_order)."""
    with open(encoders_path, "rb") as f:
        enc = pickle.load(f)
    feature_order = enc["feature_order"]

    model = _IssueTriage_NN(n_features=len(feature_order))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, feature_order


def run_inference(model, X_tensor):
    """Run model on a feature tensor, return probabilities."""
    with torch.no_grad():
        logits = model(X_tensor).squeeze(-1)
        probs = torch.sigmoid(logits)
    return probs.numpy()


def select_representative_issues(X_test, y_test, raw_df, n_total=25):
    """
    Pick ~25 issues with good coverage:
    - At least 2 from each repo
    - Mix of positive (close ≤7d) and negative labels
    - Some borderline probability cases (added after inference)
    """
    # Join test indices back to raw data for repo info
    test_indices = X_test.index
    raw_test = raw_df.loc[test_indices].copy()
    raw_test["target"] = y_test

    selected_indices = []
    repos = raw_test["repo"].unique()

    # 1) At least 2 per repo (1 positive, 1 negative if possible)
    for repo in repos:
        repo_rows = raw_test[raw_test["repo"] == repo]

        pos = repo_rows[repo_rows["target"] == 1]
        neg = repo_rows[repo_rows["target"] == 0]

        if len(pos) > 0:
            selected_indices.append(pos.sample(1, random_state=42).index[0])
        if len(neg) > 0:
            selected_indices.append(neg.sample(1, random_state=42).index[0])

    # 2) Fill remaining slots with random stratified sample
    already_selected = set(selected_indices)
    remaining = raw_test[~raw_test.index.isin(already_selected)]
    n_remaining = n_total - len(selected_indices)

    if n_remaining > 0 and len(remaining) > 0:
        # Stratified: keep roughly same positive ratio
        pos_remaining = remaining[remaining["target"] == 1]
        neg_remaining = remaining[remaining["target"] == 0]

        n_pos = max(1, int(n_remaining * y_test.mean()))
        n_neg = n_remaining - n_pos

        n_pos = min(n_pos, len(pos_remaining))
        n_neg = min(n_neg, len(neg_remaining))

        if n_pos > 0:
            selected_indices.extend(
                pos_remaining.sample(n_pos, random_state=42).index.tolist()
            )
        if n_neg > 0:
            selected_indices.extend(
                neg_remaining.sample(n_neg, random_state=42).index.tolist()
            )

    return sorted(set(selected_indices))


def main():
    print("=" * 60)
    print("  GOLDEN SET GENERATOR")
    print("=" * 60)

    # 1) Load data
    print("\nLoading data...")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["target"]
    raw_df = pd.read_parquet(PROCESSED_DIR / "cleaned_train.parquet")

    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {len(y_test)} rows")
    print(f"  raw_df: {raw_df.shape}")

    # 2) Load model
    print("\nLoading model...")
    model, feature_order = load_model(
        MODELS_DIR / "encoders.pkl",
        MODELS_DIR / "nn_weighted_model.pt",
    )
    print(f"  Features: {len(feature_order)}")

    # 3) Select representative issues
    print("\nSelecting representative issues...")
    selected_idx = select_representative_issues(X_test, y_test, raw_df)
    print(f"  Selected: {len(selected_idx)} issues")

    # 4) Run inference on selected issues
    X_selected = X_test.loc[selected_idx][feature_order]
    y_selected = y_test.loc[selected_idx]
    raw_selected = raw_df.loc[selected_idx]

    X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)
    probs = run_inference(model, X_tensor)

    # 5) Build golden set JSON
    golden_issues = []
    for i, idx in enumerate(selected_idx):
        raw_row = raw_selected.loc[idx]
        prob = float(probs[i])
        pred_class = "close_within_7d" if prob >= 0.5 else "stays_open_past_7d"

        # Confidence band (same logic as app/main.py)
        if prob >= 0.75 or prob <= 0.25:
            confidence_band = "high"
        elif prob >= 0.6 or prob <= 0.4:
            confidence_band = "medium"
        else:
            confidence_band = "low"

        issue = {
            # Raw fields (for API-level tests via /predict)
            "raw_input": {
                "title": str(raw_row.get("title", "")),
                "body": str(raw_row.get("body", ""))[:2000],  # truncate for JSON size
                "repo": str(raw_row.get("repo", "")),
                "author_association": str(raw_row.get("author_association", "")),
                "user_login": str(raw_row.get("user_login", "")),
                "created_at": str(raw_row.get("created_at", "")),
            },
            # True label
            "true_label": int(y_selected.loc[idx]),
            # Expected model outputs (frozen reference)
            "expected": {
                "predicted_class": pred_class,
                "probability": round(prob, 6),
                "confidence_band": confidence_band,
            },
            # Metadata
            "index": int(idx),
        }
        golden_issues.append(issue)

    # 6) Compute golden set metrics
    true_labels = [g["true_label"] for g in golden_issues]
    pred_labels = [1 if g["expected"]["probability"] >= 0.5 else 0 for g in golden_issues]

    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    golden_data = {
        "description": "Frozen golden test set for regression testing. DO NOT MODIFY.",
        "generated_by": "generate_golden_set.py",
        "n_issues": len(golden_issues),
        "golden_metrics": {
            "f1": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "n_positive": sum(true_labels),
            "n_negative": len(true_labels) - sum(true_labels),
        },
        "regression_thresholds": {
            "f1_floor": round(f1 - 0.05, 2),
            "note": "F1 must stay above this floor. Tolerance of 0.05 accounts for "
                    "floating-point differences across platforms.",
        },
        "issues": golden_issues,
    }

    # 7) Save
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLDEN_DIR / "golden_set.json"
    with open(output_path, "w") as f:
        json.dump(golden_data, f, indent=2, default=str)

    print(f"\n  Golden set saved: {output_path}")
    print(f"  Issues: {len(golden_issues)}")
    print(f"  Repos:  {sorted(set(g['raw_input']['repo'] for g in golden_issues))}")
    print(f"  Labels: {sum(true_labels)} positive, {len(true_labels) - sum(true_labels)} negative")
    print(f"\n  Golden metrics:")
    print(f"    F1:        {f1:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 floor:  {round(f1 - 0.05, 2)}")
    print("\nDone. This file is your frozen reference — do not edit it.")


if __name__ == "__main__":
    main()