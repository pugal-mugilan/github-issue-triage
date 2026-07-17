"""
tests/test_regression.py — Model-level regression tests.
==========================================================
Loads the trained model directly (no API), runs inference on the frozen
golden set, and asserts that metrics haven't degraded.

Catches: weight loading bugs, architecture mismatches, scaler drift,
PyTorch version changes, feature order changes.

Usage:
  pytest tests/test_regression.py -v
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_set.json"


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


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def golden_data():
    """Load the frozen golden set once for all tests in this module."""
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_and_features():
    """Load model + encoders once for all tests in this module."""
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    feature_order = enc["feature_order"]

    model = _IssueTriage_NN(n_features=len(feature_order))
    model.load_state_dict(
        torch.load(MODELS_DIR / "nn_weighted_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, feature_order


@pytest.fixture(scope="module")
def X_test():
    """Load test feature matrix."""
    return pd.read_parquet(PROCESSED_DIR / "X_test.parquet")


@pytest.fixture(scope="module")
def golden_predictions(model_and_features, X_test, golden_data):
    """Run model on golden set issues, return probabilities."""
    model, feature_order = model_and_features
    golden_indices = [issue["index"] for issue in golden_data["issues"]]

    X_golden = X_test.loc[golden_indices][feature_order]
    X_tensor = torch.tensor(X_golden.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()

    return probs


# ── Tests ────────────────────────────────────────────────────────────
class TestGoldenSetIntegrity:
    """Verify the golden set file is intact and well-formed."""

    def test_golden_file_exists(self):
        assert GOLDEN_PATH.exists(), f"Golden set not found at {GOLDEN_PATH}"

    def test_golden_has_issues(self, golden_data):
        assert len(golden_data["issues"]) >= 20, (
            f"Golden set too small: {len(golden_data['issues'])} issues (need >= 20)"
        )

    def test_golden_has_metrics(self, golden_data):
        assert "golden_metrics" in golden_data
        assert "f1" in golden_data["golden_metrics"]

    def test_golden_has_thresholds(self, golden_data):
        assert "regression_thresholds" in golden_data
        assert "f1_floor" in golden_data["regression_thresholds"]


class TestModelRegression:
    """Core regression tests — does the model still produce the same results?"""

    def test_f1_above_floor(self, golden_data, golden_predictions):
        """F1 on golden set must not drop below the frozen floor."""
        f1_floor = golden_data["regression_thresholds"]["f1_floor"]
        true_labels = [issue["true_label"] for issue in golden_data["issues"]]
        pred_labels = [1 if p >= 0.5 else 0 for p in golden_predictions]

        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        assert f1 >= f1_floor, (
            f"F1 regression! Current F1={f1:.4f}, floor={f1_floor}. "
            f"Something changed that degraded model quality."
        )

    def test_predictions_match_exactly(self, golden_data, golden_predictions):
        """On the same machine, predictions should be identical (deterministic inference)."""
        for i, issue in enumerate(golden_data["issues"]):
            expected_prob = issue["expected"]["probability"]
            actual_prob = float(golden_predictions[i])
            assert abs(actual_prob - expected_prob) < 1e-4, (
                f"Issue index {issue['index']}: "
                f"expected prob={expected_prob:.6f}, got={actual_prob:.6f}. "
                f"Prediction shifted — possible feature engineering or model change."
            )

    def test_no_class_flips(self, golden_data, golden_predictions):
        """No issue should flip its predicted class vs the golden reference."""
        flipped = []
        for i, issue in enumerate(golden_data["issues"]):
            expected_class = issue["expected"]["predicted_class"]
            actual_class = "close_within_7d" if golden_predictions[i] >= 0.5 else "stays_open_past_7d"
            if actual_class != expected_class:
                flipped.append({
                    "index": issue["index"],
                    "expected": expected_class,
                    "actual": actual_class,
                    "prob": float(golden_predictions[i]),
                })

        assert len(flipped) == 0, (
            f"{len(flipped)} issue(s) flipped class: {flipped}"
        )


class TestModelArtifacts:
    """Verify model artifacts are loadable and consistent."""

    def test_model_loads(self, model_and_features):
        model, feature_order = model_and_features
        assert model is not None
        assert len(feature_order) == 18

    def test_feature_order_matches_X_test(self, model_and_features, X_test):
        _, feature_order = model_and_features
        assert list(X_test.columns) == feature_order, (
            f"Feature order mismatch between encoders.pkl and X_test.parquet"
        )

    def test_scaler_loads(self):
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        assert hasattr(scaler, "transform"), "Scaler missing transform method"
        assert hasattr(scaler, "mean_"), "Scaler not fitted (no mean_)"

    def test_encoders_complete(self):
        with open(MODELS_DIR / "encoders.pkl", "rb") as f:
            enc = pickle.load(f)
        required_keys = ["train_repos", "train_assocs", "magnitude_cols", "feature_order"]
        for key in required_keys:
            assert key in enc, f"encoders.pkl missing key: {key}"