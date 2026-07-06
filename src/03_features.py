"""
03_features.py — Feature engineering for the issue-triage-agent capstone.

Produces a leak-free feature matrix from cleaned_train.parquet.

Outputs (data/):
  X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet

Artifacts (models/):
  scaler.pkl, encoders.pkl, author_history_lookup.parquet

Leakage rules enforced (see DECISIONS.md):
  - Train/test split BEFORE any fitting (Trap #1)
  - Scaler & one-hot vocabulary fit on train only (Trap #1)
  - Author history feature uses train rows only (Trap #2)
  - All target-derived columns dropped
  - Post-filing columns dropped (comments_count, labels)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
RANDOM_SEED = 42
TEST_FRACTION = 0.2

# Anchor all paths to the project root (parent of src/) so the script
# runs identically regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"   # cleaned_*.parquet lives here (Day 2)
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Columns we drop entirely — direct target leakage or post-filing accumulation
DROP_AS_LEAKAGE = [
    "state_reason",          # IS the target signal
    "closed_at",             # used to compute target
    "time_to_close_hours",   # LITERAL target arithmetic (closed_at - created_at) — must drop
    "comments",              # post-filing accumulation — ADR-009
    "labels",                # post-filing maintainer triage — ADR-010
    "n_labels",              # post-filing — same reasoning as labels
    "n_assignees",           # post-filing maintainer triage — ADR-011
    "has_milestone",         # post-filing maintainer triage — ADR-011
    "user_type",             # constant ("User") after Day 2 bot filter; no signal
    "issue_id",              # identifier, no signal
    "number",                # identifier, no signal
]


# =====================================================================
# 1) LOAD
# =====================================================================
print(f"Loading {PROCESSED_DIR / 'cleaned_train.parquet'}...")
df = pd.read_parquet(PROCESSED_DIR / "cleaned_train.parquet")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
print(f"  {len(df):,} rows, {df['target'].mean():.1%} positive\n")


# =====================================================================
# 2) SPLIT FIRST — before any fitting whatsoever
# =====================================================================
y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_FRACTION,
    random_state=RANDOM_SEED,
    stratify=y,
)
print(f"Split (stratified, seed={RANDOM_SEED}):")
print(f"  train: {len(X_train):,} rows ({y_train.mean():.1%} positive)")
print(f"  test:  {len(X_test):,} rows ({y_test.mean():.1%} positive)\n")


# =====================================================================
# 3) DETERMINISTIC FEATURES — text + time
#    Pure row functions. No fitting. Safe to apply to train and test
#    identically because they don't learn anything from the data.
# =====================================================================
def add_text_features(d):
    d = d.copy()
    title = d["title"].fillna("")
    body = d["body"].fillna("")
    d["title_length"] = title.str.len()
    d["body_length"] = body.str.len()
    d["body_word_count"] = body.str.split().str.len().fillna(0).astype(int)
    d["body_has_code_block"] = body.str.contains("```", regex=False).astype(int)
    d["body_has_stacktrace"] = body.str.contains(r"Traceback|\.py:\d+", regex=True).astype(int)
    d["body_has_url"] = body.str.contains(r"https?://", regex=True).astype(int)
    return d


def add_time_features(d):
    d = d.copy()
    ts = pd.to_datetime(d["created_at"], utc=True)
    d["hour_of_day"] = ts.dt.hour
    d["day_of_week"] = ts.dt.dayofweek
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    d["month"] = ts.dt.month
    return d


print("Text features...")
X_train = add_text_features(X_train)
X_test = add_text_features(X_test)

print("Time features...")
X_train = add_time_features(X_train)
X_test = add_time_features(X_test)


# =====================================================================
# 4) ONE-HOT ENCODING — fit vocabulary on TRAIN only
# =====================================================================
print("\nOne-hot encoding (vocabulary from TRAIN only)...")

train_repos = sorted(X_train["repo"].unique())
train_assocs = sorted(X_train["author_association"].fillna("UNKNOWN").unique())
print(f"  repos:  {train_repos}")
print(f"  assocs: {train_assocs}")


def apply_one_hot(d, repos, assocs):
    d = d.copy()
    for r in repos:
        d[f"repo_{r}"] = (d["repo"] == r).astype(int)
    d["author_association"] = d["author_association"].fillna("UNKNOWN")
    for a in assocs:
        d[f"aa_{a}"] = (d["author_association"] == a).astype(int)
    return d


X_train = apply_one_hot(X_train, train_repos, train_assocs)
X_test = apply_one_hot(X_test, train_repos, train_assocs)


# =====================================================================
# 5) AUTHOR HISTORY FEATURE — train-only, with merge_asof lookup for test
#    For each row, "author_prior_count" = number of issues this author
#    filed earlier in the TRAINING set. Test rows look up the training
#    side at their own created_at. No test labels enter the feature.
# =====================================================================
print("\nAuthor history (train-only)...")

# (a) Train: sort by time, then cumcount within each author = count of prior train rows
train_sorted = X_train.sort_values("created_at").copy()
train_sorted["author_prior_count"] = train_sorted.groupby("user_login").cumcount()
# "count_through" = cumcount + 1 = total train rows of this author up to AND including this row.
# This is the value we read from the lookup for test rows.
train_sorted["author_count_through"] = train_sorted["author_prior_count"] + 1

# Attach the train feature back to X_train by joining on index
X_train = X_train.join(train_sorted[["author_prior_count"]])

# (b) Test: for each test row, find the most recent prior train row of the same author
lookup = (
    train_sorted[["user_login", "created_at", "author_count_through"]]
    .sort_values("created_at")
)
test_sorted = (
    X_test.sort_values("created_at")
    .reset_index()
    .rename(columns={"index": "_orig_idx"})
)

merged = pd.merge_asof(
    test_sorted,
    lookup,
    on="created_at",
    by="user_login",
    direction="backward",
)
# Authors never seen in training → 0 prior issues
merged["author_prior_count"] = merged["author_count_through"].fillna(0).astype(int)

# Reattach to X_test, preserving original index
X_test["author_prior_count"] = merged.set_index("_orig_idx")["author_prior_count"]

print(f"  train author_prior_count: "
      f"min={X_train['author_prior_count'].min()}, "
      f"mean={X_train['author_prior_count'].mean():.1f}, "
      f"max={X_train['author_prior_count'].max()}")
print(f"  test  author_prior_count: "
      f"min={X_test['author_prior_count'].min()}, "
      f"mean={X_test['author_prior_count'].mean():.1f}, "
      f"max={X_test['author_prior_count'].max()}")


# =====================================================================
# 6) FINAL FEATURE SELECTION — drop raw columns, lock column order
# =====================================================================
print("\nSelecting final feature columns...")

raw_drop = [
    "title", "body", "user_login", "repo", "author_association", "created_at",
] + DROP_AS_LEAKAGE

X_train_features = X_train.drop(columns=[c for c in raw_drop if c in X_train.columns])
X_test_features = X_test.drop(columns=[c for c in raw_drop if c in X_test.columns])
# Enforce identical column order between train and test
X_test_features = X_test_features[X_train_features.columns]

print(f"  feature count: {X_train_features.shape[1]}")
print(f"  features: {list(X_train_features.columns)}")


# =====================================================================
# 7) SCALING — fit on TRAIN only, then transform both
#    Only magnitude columns are scaled; binary 0/1 columns don't need it.
# =====================================================================
print("\nScaling magnitude columns (fit on TRAIN only)...")

magnitude_cols = [c for c in [
    "title_length", "body_length", "body_word_count",
    "hour_of_day", "day_of_week", "month", "author_prior_count",
] if c in X_train_features.columns]
print(f"  scaling: {magnitude_cols}")

scaler = StandardScaler()
X_train_features[magnitude_cols] = scaler.fit_transform(X_train_features[magnitude_cols])
X_test_features[magnitude_cols] = scaler.transform(X_test_features[magnitude_cols])


# =====================================================================
# 8) SAVE
# =====================================================================
print("\nSaving artifacts...")

X_train_features.to_parquet(PROCESSED_DIR / "X_train.parquet")
X_test_features.to_parquet(PROCESSED_DIR / "X_test.parquet")
y_train.to_frame("target").to_parquet(PROCESSED_DIR / "y_train.parquet")
y_test.to_frame("target").to_parquet(PROCESSED_DIR / "y_test.parquet")

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(MODELS_DIR / "encoders.pkl", "wb") as f:
    pickle.dump({
        "train_repos": train_repos,
        "train_assocs": train_assocs,
        "magnitude_cols": magnitude_cols,
        "feature_order": list(X_train_features.columns),
    }, f)

# Author lookup table — used at inference time to compute author_prior_count
# for brand-new issues (the Phase 2 agent will need this).
train_sorted[["user_login", "created_at", "author_count_through"]].to_parquet(
    MODELS_DIR / "author_history_lookup.parquet"
)

print(f"\n  X_train: {X_train_features.shape}")
print(f"  X_test:  {X_test_features.shape}")
print(f"  artifacts: {PROCESSED_DIR}/, {MODELS_DIR}/")
print("\nDone.")