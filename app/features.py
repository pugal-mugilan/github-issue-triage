"""Feature engineering for inference — mirrors 03_features.py logic exactly."""

import re
from datetime import datetime

import numpy as np
import pandas as pd


def engineer_features(issue: dict, encoders: dict, scaler) -> np.ndarray:
    """
    Transform a raw issue dict into the 18-feature vector the model expects.

    Parameters
    ----------
    issue : dict with keys: title, body, repo, author_association,
            user_login, created_at
    encoders : dict from encoders.pkl (feature_order, train_repos,
               train_assocs, magnitude_cols)
    scaler : fitted StandardScaler from scaler.pkl

    Returns
    -------
    np.ndarray of shape (18,) in the exact feature_order the model expects.
    """
    title = issue.get("title", "")
    body = issue.get("body", "")
    repo = issue.get("repo", "")
    author_assoc = issue.get("author_association", "NONE")
    created_at = issue.get("created_at")

    # ── 1. Text features (6) ────────────────────────────────────
    features = {
        "title_length": len(title),
        "body_length": len(body),
        "body_word_count": len(body.split()),
        "body_has_code_block": int(bool(re.search(r"```", body))),
        "body_has_stacktrace": int(bool(re.search(
            r"Traceback|Error:|Exception:|File \"", body
        ))),
        "body_has_url": int(bool(re.search(r"https?://", body))),
    }

    # ── 2. Time features (4) ────────────────────────────────────
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    features["hour_of_day"] = created_at.hour
    features["day_of_week"] = created_at.weekday()       # 0=Mon, 6=Sun
    features["is_weekend"] = int(created_at.weekday() >= 5)
    features["month"] = created_at.month

    # ── 3. One-hot repo (3) ─────────────────────────────────────
    for r in encoders["train_repos"]:
        features[f"repo_{r}"] = int(repo == r)

    # ── 4. One-hot author association (4) ────────────────────────
    for a in encoders["train_assocs"]:
        features[f"aa_{a}"] = int(author_assoc == a)

    # ── 5. Author history (1) ───────────────────────────────────
    # At inference time we don't have the training set's history.
    # Default to 0 (first-time author). A future version could
    # maintain a live lookup table.
    features["author_prior_count"] = 0

    # ── 6. Scale magnitude columns ──────────────────────────────
    row = pd.DataFrame([features])
    row[encoders["magnitude_cols"]] = scaler.transform(
        row[encoders["magnitude_cols"]]
    )

    # ── 7. Return in exact training order ───────────────────────
    return row[encoders["feature_order"]].values[0]