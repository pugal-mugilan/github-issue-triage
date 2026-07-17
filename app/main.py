"""FastAPI entry point — GitHub Issue Triage tool contract."""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.features import engineer_features
from app.schemas import (
    IssueInput, PredictionOutput,
    BatchIssueInput, BatchPredictionOutput,
)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ── Global state (populated at startup) ──────────────────────────
model = None
scaler = None
encoders = None


# ── Model architecture (must match 06b_nn_weighted.py) ───────────

class _IssueTriage_NN(nn.Module):
    """Reconstruct the same architecture used in 06b_nn_weighted.py."""

    def __init__(self, n_features: int):
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


# ── Lifespan: load once at startup ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup."""
    global model, scaler, encoders

    # 1. Encoders (feature order, repo/assoc lists, magnitude cols)
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # 2. Scaler (fitted on training data — never re-fit)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 3. Neural network
    n_features = len(encoders["feature_order"])
    model = _IssueTriage_NN(n_features)
    model.load_state_dict(
        torch.load(MODELS_DIR / "nn_weighted_model.pt", weights_only=True)
    )
    model.eval()  # inference mode — disables dropout, batchnorm randomness

    print(f"Model loaded: {n_features} features, eval mode")
    yield  # ← server runs here, handling requests
    print("Shutting down")


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="GitHub Issue Triage Classifier",
    description=(
        "Predicts whether a GitHub issue will be resolved within 7 days. "
        "Designed as an LLM agent tool — structured input/output, "
        "confidence bands, and scope caveats on every prediction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
def root():
    """Service identity — confirms the tool is reachable."""
    return {
        "service": "github-issue-triage",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Liveness check for container orchestrators."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "model not loaded"},
        )
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    """Model metadata — lets the agent introspect what it's calling."""
    return {
        "model_version": "0.1.0",
        "architecture": "NN 18→64→32→1 (repo-weighted)",
        "features": encoders["feature_order"],
        "train_repos": encoders["train_repos"],
        "train_metrics": {
            "f1": 0.5432,
            "p_at_5_in_domain": 0.60,
            "p_at_5_ood": 0.80,
        },
        "known_limitations": [
            "P@5 in-domain missed >=0.70 threshold — feature ceiling, not model issue",
            "langchain slice has lower precision — treat predictions with reduced confidence",
            "Probability calibration not formally tested — use as ranking, not absolute likelihood",
        ],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(issue: IssueInput):
    """Predict whether a single issue will close within 7 days."""
    # 1. Engineer the 18 features from raw input
    issue_dict = issue.model_dump()
    feature_vector = engineer_features(issue_dict, encoders, scaler)

    # 2. Convert to tensor and run through model
    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    # 3. Derive confidence band
    if prob >= 0.65 or prob <= 0.35:
        band = "high"
    elif 0.45 <= prob <= 0.55:
        band = "low"
    else:
        band = "medium"

    # 4. Scope caveat
    caveat = None
    if issue.repo not in encoders["train_repos"]:
        caveat = "Repo not in training data — prediction reliability is reduced."
    elif issue.repo == "langchain-ai/langchain":
        caveat = "Langchain slice has lower precision — treat with reduced confidence."

    # 5. Return structured response
    return PredictionOutput(
        predicted_class=bool(prob >= 0.50),
        p_close_7d=round(prob, 4),
        confidence_band=band,
        top_features=encoders["feature_order"][:3],
        scope_caveat=caveat,
        model_version="0.1.0",
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchIssueInput):
    """Predict for multiple issues at once."""
    results = []
    for issue in batch.issues:
        result = predict(issue)
        results.append(result)
    return BatchPredictionOutput(predictions=results, total=len(results))