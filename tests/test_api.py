"""
tests/test_api.py — API-level contract tests.
===============================================
Spins up the FastAPI app in-process (no Docker needed), sends requests
to all endpoints, and verifies schema compliance + correct behavior.

Catches: endpoint regressions, schema changes, feature engineering drift,
error handling breakage.

Usage:
  pytest tests/test_api.py -v
"""

import json
import sys
from pathlib import Path

# Add project root to Python path so 'app' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden" / "golden_set.json"


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client (loads model once for all tests)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def golden_data():
    """Load the frozen golden set."""
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_issue(golden_data):
    """A single golden issue's raw input for quick tests."""
    return golden_data["issues"][0]["raw_input"]


# ── Health & Info Endpoints ──────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"


class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_required_fields(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        required = ["model_version", "features"]
        for field in required:
            assert field in data, f"/model/info missing field: {field}"

    def test_model_info_feature_count(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        assert len(data["features"]) == 18


# ── Single Prediction Endpoint ───────────────────────────────────────
class TestPredictEndpoint:

    def test_predict_returns_200(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        assert resp.status_code == 200

    def test_predict_response_has_required_fields(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        data = resp.json()
        required = [
            "predicted_class", "p_close_7d", "confidence_band",
            "top_features", "scope_caveat", "model_version",
        ]
        for field in required:
            assert field in data, f"/predict response missing field: {field}"

    def test_predict_class_is_valid(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        data = resp.json()
        valid_classes = {"close_within_7d", "stays_open_past_7d", True, False}
        assert data["predicted_class"] in valid_classes

    def test_predict_probability_in_range(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        data = resp.json()
        assert 0.0 <= data["p_close_7d"] <= 1.0, (
            f"Probability out of range: {data['p_close_7d']}"
        )

    def test_predict_confidence_band_is_valid(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        data = resp.json()
        valid_bands = {"low", "medium", "high"}
        assert data["confidence_band"] in valid_bands

    def test_predict_top_features_is_list(self, client, sample_issue):
        resp = client.post("/predict", json=sample_issue)
        data = resp.json()
        assert isinstance(data["top_features"], list)

    def test_predict_consistency_with_golden(self, client, golden_data):
        """Predictions via the API should agree on class direction (not exact probability)."""
        mismatches = []
        for issue in golden_data["issues"][:10]:
            resp = client.post("/predict", json=issue["raw_input"])
            assert resp.status_code == 200
            data = resp.json()

            # Map golden string labels to boolean for comparison
            expected_bool = issue["expected"]["predicted_class"] == "close_within_7d"
            actual_class = data["predicted_class"]

            # Handle both bool and string formats from the API
            if isinstance(actual_class, str):
                actual_bool = actual_class == "close_within_7d"
            else:
                actual_bool = bool(actual_class)

            if actual_bool != expected_bool:
                mismatches.append({
                    "index": issue["index"],
                    "expected": expected_bool,
                    "actual": actual_bool,
                    "api_prob": data["p_close_7d"],
                    "golden_prob": issue["expected"]["probability"],
                })

        assert len(mismatches) == 0, (
            f"API class predictions don't match golden set: {mismatches}. "
            f"Possible feature engineering drift between training and serving."
        )


# ── Batch Prediction Endpoint ────────────────────────────────────────
class TestBatchPredictEndpoint:

    def test_batch_returns_200(self, client, golden_data):
        issues = [g["raw_input"] for g in golden_data["issues"][:5]]
        resp = client.post("/predict/batch", json={"issues": issues})
        assert resp.status_code == 200

    def test_batch_returns_correct_count(self, client, golden_data):
        issues = [g["raw_input"] for g in golden_data["issues"][:5]]
        resp = client.post("/predict/batch", json={"issues": issues})
        data = resp.json()
        assert data["total"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_each_prediction_has_required_fields(self, client, golden_data):
        issues = [g["raw_input"] for g in golden_data["issues"][:3]]
        resp = client.post("/predict/batch", json={"issues": issues})
        data = resp.json()
        required = ["predicted_class", "p_close_7d", "confidence_band"]
        for pred in data["predictions"]:
            for field in required:
                assert field in pred, f"Batch prediction missing field: {field}"


# ── Error Handling ───────────────────────────────────────────────────
class TestErrorHandling:

    def test_missing_required_field_returns_422(self, client):
        """Sending an issue without 'title' should fail validation."""
        bad_input = {
            "body": "some body text",
            "repo": "pytorch/pytorch",
        }
        resp = client.post("/predict", json=bad_input)
        assert resp.status_code == 422

    def test_empty_body_returns_200(self, client, sample_issue):
        """Empty body is valid — many issues have no body."""
        issue_with_empty_body = {**sample_issue, "body": ""}
        resp = client.post("/predict", json=issue_with_empty_body)
        assert resp.status_code == 200

    def test_batch_empty_list_handling(self, client):
        """Empty batch should either return 200 with empty list or 400."""
        resp = client.post("/predict/batch", json={"issues": []})
        # Either behavior is acceptable — just shouldn't 500
        assert resp.status_code in (200, 400, 422)

    def test_invalid_json_returns_422(self, client):
        """Completely wrong structure should fail."""
        resp = client.post("/predict", json={"garbage": "data"})
        assert resp.status_code == 422

    def test_get_on_predict_returns_405(self, client):
        """GET on a POST-only endpoint should return 405 Method Not Allowed."""
        resp = client.get("/predict")
        assert resp.status_code == 405