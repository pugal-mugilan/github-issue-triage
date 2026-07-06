"""
08_ood_eval.py — Out-of-distribution evaluation
================================================
Applies the saved feature pipeline (scaler, encoders, author lookup)
to cleaned_ood.parquet (fastapi + scikit-learn), then evaluates the
repo-weighted NN on data it has never seen.

Success criterion from PROBLEM.md:
  - OOD precision@5 ≥ 0.55

Key OOD differences from in-domain:
  - repo one-hot columns will be ALL ZEROS (these repos weren't in training)
  - author_prior_count will mostly be 0 (different contributor pools)
  - community patterns (body length, code blocks, etc.) may differ
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Load saved artifacts ─────────────────────────────────────────────
with open(MODELS_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

train_repos    = encoders["train_repos"]
train_assocs   = encoders["train_assocs"]
magnitude_cols = encoders["magnitude_cols"]
feature_order  = encoders["feature_order"]

author_lookup = pd.read_parquet(MODELS_DIR / "author_history_lookup.parquet")

print("Loaded artifacts:")
print(f"  repos vocabulary:  {train_repos}")
print(f"  assocs vocabulary: {train_assocs}")
print(f"  feature order:     {feature_order}")
print(f"  author lookup:     {len(author_lookup):,} rows\n")

# ── Load OOD data ────────────────────────────────────────────────────
df = pd.read_parquet(PROCESSED / "cleaned_ood.parquet")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
print(f"OOD data: {len(df):,} rows, {df['target'].mean():.1%} positive rate")

# Show per-repo breakdown
for repo in sorted(df["repo"].unique()):
    mask = df["repo"] == repo
    n = mask.sum()
    pos = df.loc[mask, "target"].sum()
    print(f"  {repo}: {n} rows, {pos} positives ({pos/n:.1%})")

y_ood = df["target"]
X_ood = df.drop(columns=["target"])

# ── Apply feature engineering (same as 03_features.py) ───────────────

# 3a) Text features — pure row functions, no fitting
print("\nEngineering features...")
X = X_ood.copy()
title = X["title"].fillna("")
body  = X["body"].fillna("")
X["title_length"]        = title.str.len()
X["body_length"]         = body.str.len()
X["body_word_count"]     = body.str.split().str.len().fillna(0).astype(int)
X["body_has_code_block"] = body.str.contains("```", regex=False).astype(int)
X["body_has_stacktrace"] = body.str.contains(r"Traceback|\.py:\d+", regex=True).astype(int)
X["body_has_url"]        = body.str.contains(r"https?://", regex=True).astype(int)

# 3b) Time features
ts = pd.to_datetime(X["created_at"], utc=True)
X["hour_of_day"] = ts.dt.hour
X["day_of_week"] = ts.dt.dayofweek
X["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
X["month"]       = ts.dt.month

# 3c) One-hot encoding — using TRAINING vocabulary
#     OOD repos (fastapi, scikit-learn) won't match any training repo,
#     so all repo_ columns will be 0. This is correct and expected.
for r in train_repos:
    X[f"repo_{r}"] = (X["repo"] == r).astype(int)
X["author_association"] = X["author_association"].fillna("UNKNOWN")
for a in train_assocs:
    X[f"aa_{a}"] = (X["author_association"] == a).astype(int)

# 3d) Author history — lookup against TRAINING data only
X_sorted = X.sort_values("created_at").reset_index().rename(columns={"index": "_orig_idx"})
lookup_sorted = author_lookup.sort_values("created_at")

merged = pd.merge_asof(
    X_sorted,
    lookup_sorted,
    on="created_at",
    by="user_login",
    direction="backward",
)
merged["author_prior_count"] = merged["author_count_through"].fillna(0).astype(int)
X["author_prior_count"] = merged.set_index("_orig_idx")["author_prior_count"]

# 3e) Drop raw columns, enforce training feature order
DROP_AS_LEAKAGE = [
    "state_reason", "closed_at", "time_to_close_hours", "comments",
    "labels", "n_labels", "n_assignees", "has_milestone", "user_type",
    "issue_id", "number",
]
raw_drop = [
    "title", "body", "user_login", "repo", "author_association", "created_at",
] + DROP_AS_LEAKAGE

X_features = X.drop(columns=[c for c in raw_drop if c in X.columns])

# Add any missing columns as zeros (OOD won't have some training columns)
for col in feature_order:
    if col not in X_features.columns:
        X_features[col] = 0

# Enforce exact column order from training
X_features = X_features[feature_order]

# 3f) Scale magnitude columns using TRAINING scaler
X_features[magnitude_cols] = scaler.transform(X_features[magnitude_cols])

print(f"  Final feature matrix: {X_features.shape}")
print(f"  Columns match training: {list(X_features.columns) == feature_order}")

# ── Load model ───────────────────────────────────────────────────────
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

n_features = len(feature_order)
model = IssueTriage_NN(n_features)
state_dict = torch.load(MODELS_DIR / "nn_weighted_model.pt", weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"\nLoaded model: {sum(p.numel() for p in model.parameters()):,} parameters")

# ── Score OOD data ───────────────────────────────────────────────────
X_ood_t = torch.tensor(X_features.values, dtype=torch.float32)

with torch.no_grad():
    logits = model(X_ood_t)
    probs  = torch.sigmoid(logits).squeeze().numpy()

y_pred = (probs >= 0.5).astype(int)
y_true = y_ood.values

# ── Precision@K function ─────────────────────────────────────────────
def precision_at_k(y_true, probs, k):
    if len(y_true) < k:
        return None
    top_k_idx = np.argsort(probs)[::-1][:k]
    return y_true[top_k_idx].sum() / k

# ── Overall OOD metrics ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("OOD EVALUATION — Overall (fastapi + scikit-learn combined)")
print("=" * 60)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 (pos):  {f1:.4f}")

for k in [5, 10, 20]:
    p = precision_at_k(y_true, probs, k)
    if p is not None:
        print(f"  P@{k:<3d} =    {p:.4f}  ({int(p * k)}/{k} correct)")

# ── Per-repo OOD metrics ─────────────────────────────────────────────
ood_repos = sorted(df["repo"].unique())

print(f"\n{'=' * 60}")
print("OOD EVALUATION — Per Repo")
print("=" * 60)

for repo in ood_repos:
    mask = (df["repo"] == repo).values
    repo_y    = y_true[mask]
    repo_pred = y_pred[mask]
    repo_prob = probs[mask]
    n_total   = mask.sum()
    n_pos     = repo_y.sum()

    repo_f1 = f1_score(repo_y, repo_pred, zero_division=0)

    print(f"\n  {repo}")
    print(f"  {'─' * 50}")
    print(f"  Rows: {n_total} | Positives: {int(n_pos)} ({n_pos/n_total:.1%})")
    print(f"  F1:   {repo_f1:.4f}")

    for k in [5, 10]:
        p = precision_at_k(repo_y, repo_prob, k)
        if p is not None:
            print(f"  P@{k:<3d} = {p:.4f}  ({int(p * k)}/{k} correct)")
        else:
            print(f"  P@{k:<3d} = N/A (only {n_total} rows)")

# ── Probability diagnostic ───────────────────────────────────────────
print(f"\n{'=' * 60}")
print("PROBABILITY DISTRIBUTION — OOD")
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
    actual = "YES" if y_true[idx] == 1 else "NO"
    repo   = df.iloc[idx]["repo"]
    print(f"    #{rank:2d}  prob={probs[idx]:.4f}  actual={actual}  repo={repo}")

# ── Success criteria check ───────────────────────────────────────────
ood_p5 = precision_at_k(y_true, probs, 5)

print(f"\n{'=' * 60}")
print("SUCCESS CRITERIA CHECK (from PROBLEM.md)")
print("=" * 60)
print(f"  OOD P@5:  {ood_p5:.4f}  "
      f"{'PASS' if ood_p5 >= 0.55 else 'FAIL'}  (threshold: >= 0.55)")
print()