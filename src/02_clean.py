"""
02_clean.py — Load raw GitHub issue JSON, apply cleaning pipeline,
save train + held-out OOD datasets as Parquet.

Pipeline per repo:
  1. Drop pull requests (PR != Issue lifecycle)
  2. Convert datetime columns
  3. Drop rows outside the since_date window
  4. Drop rows opened within last 168h (label unobservable)
  5. Drop bot-authored rows
  6. Drop rows with null state_reason (no labelable outcome)
  7. Dedupe by issue_id
  8. Compute target = (state_reason == "completed") & (close_time <= 168h)
  9. Type discipline (categoricals)

Output:
  data/processed/cleaned_train.parquet  (3 in-domain repos)
  data/processed/cleaned_ood.parquet    (2 held-out repos)
  data/processed/_clean_metadata.json   (audit trail)
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ---------- Config ----------

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_REPOS = {
    "huggingface/transformers",
    "pytorch/pytorch",
    "langchain-ai/langchain",
}
OOD_REPOS = {
    "fastapi/fastapi",
    "scikit-learn/scikit-learn",
}

HORIZON_HOURS = 168

KNOWN_BOT_SUBSTRINGS = {"dependabot", "github-actions", "renovate", "pre-commit-ci"}

# Anchor time math on the scrape moment, not on now
with open(RAW_DIR / "_metadata.json") as f:
    METADATA = json.load(f)
SCRAPED_AT = datetime.fromisoformat(METADATA["scraped_at"])
SINCE_DATE = datetime.fromisoformat(METADATA["since_date"])
OBSERVABILITY_CUTOFF = SCRAPED_AT - timedelta(hours=HORIZON_HOURS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("clean")


# ---------- Field extraction ----------

def extract_fields(raw_item, repo):
    """Pluck and flatten the fields we care about from a raw issue dict."""
    user = raw_item.get("user") or {}
    return {
        "repo": repo,
        "issue_id": str(raw_item.get("id")),
        "number": raw_item.get("number"),
        "title": raw_item.get("title") or "",
        "body": raw_item.get("body") or "",
        "created_at": raw_item.get("created_at"),
        "closed_at": raw_item.get("closed_at"),
        "state_reason": raw_item.get("state_reason"),
        "comments": raw_item.get("comments", 0),
        "labels": [lbl["name"] for lbl in raw_item.get("labels") or []],
        "n_labels": len(raw_item.get("labels") or []),
        "user_login": user.get("login"),
        "user_type": user.get("type"),
        "author_association": raw_item.get("author_association"),
        "n_assignees": len(raw_item.get("assignees") or []),
        "has_milestone": raw_item.get("milestone") is not None,
        "is_pull_request": "pull_request" in raw_item,
    }


# ---------- Bot detection ----------

def is_bot(row):
    """Three signals OR'd together: explicit type, [bot] suffix, known names."""
    if row["user_type"] == "Bot":
        return True
    login = (row["user_login"] or "").lower()
    if login.endswith("[bot]"):
        return True
    if any(b in login for b in KNOWN_BOT_SUBSTRINGS):
        return True
    return False


# ---------- Per-repo load + clean ----------

def load_and_extract(repo_owner, repo_name):
    repo_key = f"{repo_owner}/{repo_name}"
    path = RAW_DIR / f"{repo_owner}__{repo_name}.json"
    with open(path) as f:
        raw = json.load(f)
    log.info(f"Loaded {repo_key}: {len(raw):,} raw items")
    rows = [extract_fields(item, repo_key) for item in raw]
    return pd.DataFrame(rows)


def clean(df, repo_key):
    """Apply the full cleaning pipeline to one repo's DataFrame."""
    counts = {"raw": len(df)}

    # 1. Drop pull requests
    df = df[~df["is_pull_request"]].copy()
    counts["after_drop_prs"] = len(df)

    # 2. Type conversion for dates (must happen before any time filter)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)

    # 3. Drop rows created before the since_date window
    #    (API's `since` param filters by updated_at, not created_at)
    df = df[df["created_at"] >= SINCE_DATE].copy()
    counts["after_window"] = len(df)

    # 4. 168h observability filter — drop issues whose 7-day window hasn't completed
    df = df[df["created_at"] <= OBSERVABILITY_CUTOFF].copy()
    counts["after_observability"] = len(df)

    # 5. Drop bots
    df["is_bot"] = df.apply(is_bot, axis=1)
    df = df[~df["is_bot"]].copy()
    counts["after_drop_bots"] = len(df)

    # 6. Drop rows with null state_reason (no labelable outcome)
    df = df.dropna(subset=["state_reason"]).copy()
    counts["after_drop_null_state_reason"] = len(df)

    # 7. Dedupe by issue_id (defensive — shouldn't be many)
    n_before = len(df)
    df = df.drop_duplicates(subset=["issue_id"]).copy()
    counts["dedupe_dropped"] = n_before - len(df)
    counts["after_dedupe"] = len(df)

    # 8. Compute target
    df["time_to_close_hours"] = (
        (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600
    )
    df["target"] = (
        (df["state_reason"] == "completed")
        & (df["time_to_close_hours"] <= HORIZON_HOURS)
    ).astype(int)
    counts["final"] = len(df)
    counts["positive_rate"] = round(float(df["target"].mean()), 4) if len(df) else 0.0

    # 9. Type discipline
    df["user_type"] = df["user_type"].astype("category")
    df["author_association"] = df["author_association"].astype("category")
    df["state_reason"] = df["state_reason"].astype("category")
    df["repo"] = df["repo"].astype("category")

    # Drop intermediate flags
    df = df.drop(columns=["is_pull_request", "is_bot"])

    log.info(f"  {repo_key} funnel: {counts}")
    return df, counts


def main():
    train_frames = []
    ood_frames = []
    all_counts = {}

    REPOS = [
        ("huggingface", "transformers"),
        ("pytorch", "pytorch"),
        ("langchain-ai", "langchain"),
        ("fastapi", "fastapi"),
        ("scikit-learn", "scikit-learn"),
    ]

    for repo_owner, repo_name in REPOS:
        repo_key = f"{repo_owner}/{repo_name}"
        df_raw = load_and_extract(repo_owner, repo_name)
        df_clean, counts = clean(df_raw, repo_key)
        all_counts[repo_key] = counts

        if repo_key in TRAIN_REPOS:
            train_frames.append(df_clean)
        elif repo_key in OOD_REPOS:
            ood_frames.append(df_clean)
        else:
            raise ValueError(f"{repo_key} not in TRAIN or OOD set")

    train_df = pd.concat(train_frames, ignore_index=True)
    ood_df = pd.concat(ood_frames, ignore_index=True)

    train_path = OUT_DIR / "cleaned_train.parquet"
    ood_path = OUT_DIR / "cleaned_ood.parquet"
    train_df.to_parquet(train_path, index=False)
    ood_df.to_parquet(ood_path, index=False)

    log.info("")
    log.info(f"TRAIN  {len(train_df):>6,} rows  ·  {train_df['target'].mean():.1%} positive  ·  {train_path}")
    log.info(f"OOD    {len(ood_df):>6,} rows  ·  {ood_df['target'].mean():.1%} positive  ·  {ood_path}")

    summary = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "scrape_metadata": METADATA,
        "horizon_hours": HORIZON_HOURS,
        "observability_cutoff": OBSERVABILITY_CUTOFF.isoformat(),
        "per_repo_funnel": all_counts,
        "train_rows": len(train_df),
        "ood_rows": len(ood_df),
        "train_positive_rate": round(float(train_df["target"].mean()), 4),
        "ood_positive_rate": round(float(ood_df["target"].mean()), 4),
    }
    with open(OUT_DIR / "_clean_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {OUT_DIR / '_clean_metadata.json'}")


if __name__ == "__main__":
    main()