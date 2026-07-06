"""
05a_seed_check.py — Multi-seed stability check
================================================
Runs XGBoost 5 times with different random seeds.
Checks whether the transformers F1 gate failure is
real (consistent) or noise (varies across seeds).

Quick diagnostic — not a production script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb

# ── Load data ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"

X_train_full = pd.read_parquet(PROCESSED / "X_train.parquet")
X_test       = pd.read_parquet(PROCESSED / "X_test.parquet")
y_train_full = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
y_test       = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

# ── Repo masks (precompute once) ────────────────────────────────────
REPO_COLS  = [c for c in X_test.columns if c.startswith("repo_")]
REPO_NAMES = [c.replace("repo_", "") for c in REPO_COLS]

repo_masks = {}
for col, name in zip(REPO_COLS, REPO_NAMES):
    repo_masks[name] = X_test[col] == 1

# ── LR bars to compare against ──────────────────────────────────────
LR_BARS = {
    "huggingface/transformers": 0.5913,
    "langchain-ai/langchain":  0.0000,
    "pytorch/pytorch":         0.5032,
}

# ── Run 5 seeds ──────────────────────────────────────────────────────
SEEDS = [42, 123, 256, 789, 1001]

results = []

for seed in SEEDS:
    # Fresh val split each time (different seed → different val rows)
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=seed, stratify=y_train_full
    )

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    y_pred = model.predict(X_test)

    row = {"seed": seed, "best_round": model.best_iteration}
    row["overall_f1"] = f1_score(y_test, y_pred, zero_division=0)

    for name, mask in repo_masks.items():
        row[f"f1_{name}"] = f1_score(y_test[mask], y_pred[mask], zero_division=0)

    results.append(row)

    # Quick one-line progress
    tf1 = row["f1_huggingface/transformers"]
    bar = LR_BARS["huggingface/transformers"]
    status = "✓" if tf1 > bar else "✗"
    print(f"  seed={seed:>5}  round={row['best_round']:>3}  "
          f"overall={row['overall_f1']:.4f}  "
          f"transformers={tf1:.4f} {status}  "
          f"langchain={row['f1_langchain-ai/langchain']:.4f}  "
          f"pytorch={row['f1_pytorch/pytorch']:.4f}")

# ── Summary ──────────────────────────────────────────────────────────
df = pd.DataFrame(results)
print()
print("=" * 70)
print("SUMMARY ACROSS 5 SEEDS")
print("=" * 70)

for col in ["overall_f1", "f1_huggingface/transformers",
            "f1_langchain-ai/langchain", "f1_pytorch/pytorch"]:
    vals = df[col]
    label = col.replace("f1_", "").replace("overall_f1", "overall")
    print(f"  {label:30s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}")

# ── Transformers gate verdict ────────────────────────────────────────
tf1_vals = df["f1_huggingface/transformers"]
bar = LR_BARS["huggingface/transformers"]
beats = (tf1_vals > bar).sum()
print()
print(f"Transformers gate (bar={bar:.4f}): passed {beats}/5 seeds")
if beats >= 3:
    print("→ Gap is NOISE. XGBoost is statistically tied with LR on transformers.")
    print("  Accept the model. Document in ADR DL-012.")
elif beats == 0:
    print("→ Gap is REAL. XGBoost consistently underperforms LR on transformers.")
    print("  Consider tuning or investigating the transformers data slice.")
else:
    print("→ BORDERLINE. Mixed results — consider more seeds or bootstrap CI.")