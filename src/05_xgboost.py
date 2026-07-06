"""
05_xgboost.py — XGBoost classifier for issue-triage
====================================================
Trains a gradient-boosted tree ensemble on the in-domain data.

Key design choices:
  - Validation split carved from training data (test stays untouched)
  - scale_pos_weight to handle 63/37 class imbalance
  - Per-repo F1 breakdown to catch pytorch-dominance masking (DL-012)
  - Early stopping on validation log-loss to prevent overfitting

Inputs:  data/processed/{X_train, X_test, y_train, y_test}.parquet
Outputs: models/xgb_model.json, printed metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report
)
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Load pre-split data ─────────────────────────────────────────────
X_train_full = pd.read_parquet(PROCESSED / "X_train.parquet")
X_test       = pd.read_parquet(PROCESSED / "X_test.parquet")
y_train_full = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
y_test       = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

print(f"Full train: {len(X_train_full):,} | Test: {len(X_test):,}")

# ── Validation split (from training data only) ──────────────────────
# WHY: If we tune hyperparameters by looking at test-set metrics,
# we're indirectly fitting to test. The validation set is a safe
# target for tuning — test stays sealed until the very end.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Positive rate — train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")
print()

# ── Compute scale_pos_weight ─────────────────────────────────────────
# Ratio of negatives to positives in training set.
# Tells XGBoost: "a missed positive hurts this many times more."
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw = neg_count / pos_count
print(f"scale_pos_weight = {neg_count} / {pos_count} = {spw:.2f}")
print()

# ── Train XGBoost ────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,           # max trees (early stopping may use fewer)
    max_depth=4,                # shallow trees — boosting builds depth across rounds
    learning_rate=0.1,          # each tree corrects 10% of remaining error
    scale_pos_weight=spw,       # class imbalance fix
    eval_metric="logloss",      # monitor validation log-loss for early stopping
    early_stopping_rounds=20,   # stop if val loss doesn't improve for 20 rounds
    random_state=42,
    n_jobs=-1,                  # use all CPU cores
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # early stopping watches this
    verbose=False,              # suppress per-round logging
)

best_round = model.best_iteration
print(f"Training complete. Best round: {best_round} / 300")
print()

# ── Save model ───────────────────────────────────────────────────────
model_path = MODELS_DIR / "xgb_model.json"
model.save_model(str(model_path))
print(f"Model saved → {model_path}")
print()

# ── Evaluation helpers ───────────────────────────────────────────────
REPO_COLS  = [c for c in X_test.columns if c.startswith("repo_")]
REPO_NAMES = [c.replace("repo_", "") for c in REPO_COLS]

def per_repo_f1(y_true, y_pred, X):
    results = {}
    for col, name in zip(REPO_COLS, REPO_NAMES):
        mask = X[col] == 1
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        n  = mask.sum()
        results[name] = (f1, n)
    return results

def evaluate(name, y_true, y_pred, X):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    repo_f1s = per_repo_f1(y_true, y_pred, X)

    print(f"{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 (pos):  {f1:.4f}")
    for repo, (rf1, n) in repo_f1s.items():
        print(f"    └─ {repo:30s} F1 = {rf1:.4f}  (n={n})")
    print()
    return f1

# ── Evaluate on validation set ───────────────────────────────────────
y_pred_val = model.predict(X_val)
evaluate("XGBoost — Validation set", y_val, y_pred_val, X_val)

# ── Evaluate on test set ─────────────────────────────────────────────
y_pred_test = model.predict(X_test)
f1_test = evaluate("XGBoost — Test set (sealed until now)", y_test, y_pred_test, X_test)

# ── Full classification report on test ───────────────────────────────
print("Full classification report (test):")
print(classification_report(y_test, y_pred_test, target_names=["NOT resolved", "Resolved <168h"]))

# ── Baseline comparison ──────────────────────────────────────────────
print("=" * 60)
print("BASELINE COMPARISON (test set, positive-class F1)")
print("=" * 60)
print(f"  Brick (majority):       0.0000")
print(f"  Coin  (stratified):     0.3737")
print(f"  Logistic Regression:    0.5122")
print(f"  XGBoost:                {f1_test:.4f}  {'✓ BEATS ALL' if f1_test > 0.5122 else '✗ Does not beat LR'}")
print()

# ── DL-012 per-repo gate ─────────────────────────────────────────────
LR_REPO_BARS = {
    "huggingface/transformers": 0.5913,
    "langchain-ai/langchain":  0.0000,   # LR failed here; coin got 0.2857
    "pytorch/pytorch":         0.5032,
}
print("Per-repo DL-012 gate:")
repo_f1s = per_repo_f1(y_test, y_pred_test, X_test)
all_pass = True
for repo, (rf1, n) in repo_f1s.items():
    bar = LR_REPO_BARS.get(repo, 0.0)
    passed = rf1 > bar
    if not passed:
        all_pass = False
    print(f"  {repo:30s}  F1={rf1:.4f}  bar={bar:.4f}  {'✓' if passed else '✗ FAIL'}")
print(f"\nOverall DL-012: {'✓ ALL GATES PASSED' if all_pass else '✗ SOME GATES FAILED'}")