"""
test_pipeline.py — Smoke test for the trained pipeline
=======================================================
Loads the saved model, scaler, and encoders, runs a single
fake issue through the prediction path, and asserts the output
schema is correct.

Usage:
  python test_pipeline.py

This test verifies:
  1. All saved artifacts load without error
  2. The model accepts input shaped like real features
  3. The output is a probability between 0 and 1
  4. The binary prediction is 0 or 1
  5. metrics.json exists and has required keys
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅  {name}")
        passed += 1
    else:
        print(f"  ❌  {name}  — {detail}")
        failed += 1


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


def main():
    global passed, failed
    print("=" * 60)
    print("  SMOKE TEST — Issue Triage Pipeline")
    print("=" * 60 + "\n")

    # ── Test 1: Artifacts exist ──────────────────────────────────────
    print("Artifact existence:")
    check("nn_weighted_model.pt exists", (MODELS_DIR / "nn_weighted_model.pt").exists())
    check("scaler.pkl exists", (MODELS_DIR / "scaler.pkl").exists())
    check("encoders.pkl exists", (MODELS_DIR / "encoders.pkl").exists())
    check("author_history_lookup.parquet exists",
          (MODELS_DIR / "author_history_lookup.parquet").exists())
    check("metrics.json exists", (MODELS_DIR / "metrics.json").exists())

    # ── Test 2: Encoders load and contain required keys ──────────────
    print("\nEncoder contents:")
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    required_keys = {"train_repos", "train_assocs", "magnitude_cols", "feature_order"}
    check("encoders has required keys",
          required_keys.issubset(encoders.keys()),
          f"missing: {required_keys - encoders.keys()}")

    feature_order = encoders["feature_order"]
    n_features = len(feature_order)
    check(f"feature_order has {n_features} features", n_features > 0)

    # ── Test 3: Scaler loads ─────────────────────────────────────────
    print("\nScaler:")
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    check("scaler loads", hasattr(scaler, "transform"))
    check("scaler n_features matches magnitude_cols",
          scaler.n_features_in_ == len(encoders["magnitude_cols"]),
          f"scaler expects {scaler.n_features_in_}, got {len(encoders['magnitude_cols'])}")

    # ── Test 4: Model loads and accepts correct input shape ──────────
    print("\nModel:")
    model = IssueTriage_NN(n_features)
    state_dict = torch.load(MODELS_DIR / "nn_weighted_model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    check("model loads from state_dict", True)

    param_count = sum(p.numel() for p in model.parameters())
    check(f"model has {param_count:,} parameters", param_count > 0)

    # ── Test 5: Forward pass with fake input ─────────────────────────
    print("\nForward pass (fake input):")
    fake_input = torch.randn(1, n_features)

    with torch.no_grad():
        logits = model(fake_input)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)

    check("logits shape is (1, 1)", logits.shape == (1, 1),
          f"got {logits.shape}")
    check(f"probability is between 0 and 1: {prob:.4f}",
          0.0 <= prob <= 1.0)
    check(f"binary prediction is 0 or 1: {pred}",
          pred in (0, 1))

    # ── Test 6: metrics.json structure ───────────────────────────────
    print("\nMetrics file:")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    required_top = {"pipeline_version", "seed", "training", "precision_at_k", "ood",
                    "success_criteria"}
    check("metrics.json has required top-level keys",
          required_top.issubset(metrics.keys()),
          f"missing: {required_top - metrics.keys()}")

    check("training.f1 is a number",
          isinstance(metrics.get("training", {}).get("f1"), (int, float)),
          f"got {type(metrics.get('training', {}).get('f1'))}")

    check("success_criteria present",
          "success_criteria" in metrics and len(metrics["success_criteria"]) == 3,
          f"got {metrics.get('success_criteria')}")

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("  All smoke tests passed. ✅\n")
        sys.exit(0)


if __name__ == "__main__":
    main()