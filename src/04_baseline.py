"""
04_baseline.py — Three-tier baseline evaluation
================================================
Tier 1: Majority-class predictor (the "brick")
Tier 2: Stratified random predictor (the "biased coin")
Tier 3: Logistic regression (simplest real model)

All evaluated on:
  - Overall positive-class F1
  - Per-repo positive-class F1 (pytorch / transformers / langchain)

Inputs:  data/processed/{X_train, X_test, y_train, y_test}.parquet
Outputs: printed comparison table (no saved artifacts yet)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── Load pre-split data ─────────────────────────────────────────────
X_train = pd.read_parquet(PROCESSED / "X_train.parquet")
X_test  = pd.read_parquet(PROCESSED / "X_test.parquet")
y_train = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
y_test  = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"Positive rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")
print()

# ── Identify repo columns for per-repo breakdown ────────────────────
# One-hot columns from 03_features.py are named repo_<name>
REPO_COLS = [c for c in X_test.columns if c.startswith("repo_")]
REPO_NAMES = [c.replace("repo_", "") for c in REPO_COLS]

def per_repo_f1(y_true, y_pred, X):
    """Compute positive-class F1 for each repo slice."""
    results = {}
    for col, name in zip(REPO_COLS, REPO_NAMES):
        mask = X[col] == 1
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        results[name] = f1
    return results

def evaluate(name, y_pred):
    """Print overall + per-repo metrics for one baseline."""
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    repo_f1s = per_repo_f1(y_test, y_pred, X_test)

    print(f"{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 (pos):  {f1:.4f}")
    for repo, rf1 in repo_f1s.items():
        print(f"    └─ {repo:20s} F1 = {rf1:.4f}")
    print()
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, **{f"f1_{r}": v for r, v in repo_f1s.items()}}


# ═══════════════════════════════════════════════════════════════════
# TIER 1 — Majority-class predictor ("the brick")
# ═══════════════════════════════════════════════════════════════════
# strategy="most_frequent" → always predicts the majority class (0)
brick = DummyClassifier(strategy="most_frequent", random_state=42)
brick.fit(X_train, y_train)
y_pred_brick = brick.predict(X_test)

row_brick = evaluate("Tier 1: Majority-class (brick)", y_pred_brick)


# ═══════════════════════════════════════════════════════════════════
# TIER 2 — Stratified random predictor ("the biased coin")
# ═══════════════════════════════════════════════════════════════════
# strategy="stratified" → predicts each class with probability
# matching its training-set frequency (≈37% YES, ≈63% NO)
coin = DummyClassifier(strategy="stratified", random_state=42)
coin.fit(X_train, y_train)
y_pred_coin = coin.predict(X_test)

row_coin = evaluate("Tier 2: Stratified random (coin)", y_pred_coin)


# ═══════════════════════════════════════════════════════════════════
# TIER 3 — Logistic Regression (simplest real model)
# ═══════════════════════════════════════════════════════════════════
# max_iter=1000: give it room to converge
# Data is already scaled (scaler.pkl fit on train in 03_features.py)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

row_lr = evaluate("Tier 3: Logistic Regression", y_pred_lr)


# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════
summary = pd.DataFrame([row_brick, row_coin, row_lr])
print("=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)
print(summary.to_string(index=False, float_format="{:.4f}".format))
print()
print("Any model trained tomorrow must beat ALL rows above on F1 (pos),")
print("both overall AND per-repo, to be accepted (DL-012).")