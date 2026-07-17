"""Pydantic schemas — the tool contract for the GitHub Issue Triage classifier."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ──────────────────────────────────────────────

class IssueInput(BaseModel):
    """A single GitHub issue, as the agent would see it at filing time."""
    title: str = Field(..., description="Issue title text")
    body: str = Field("", description="Issue body/description (may be empty)")
    repo: str = Field(..., description="Full repo name, e.g. 'pytorch/pytorch'")
    author_association: str = Field(
        ...,
        description="Author's relationship to the repo: COLLABORATOR | CONTRIBUTOR | MEMBER | NONE"
    )
    user_login: str = Field(..., description="GitHub username of the issue author")
    created_at: datetime = Field(..., description="Issue creation timestamp, ISO 8601 UTC")


class BatchIssueInput(BaseModel):
    """A batch of issues for bulk prediction."""
    issues: list[IssueInput] = Field(..., description="List of issues to classify", min_length=1)


# ── Response schemas ─────────────────────────────────────────────

class PredictionOutput(BaseModel):
    """Prediction for a single issue — designed for LLM agent consumption."""
    predicted_class: bool = Field(
        ..., description="True = likely resolved within 7 days, False = unlikely"
    )
    p_close_7d: float = Field(
        ..., description="Probability of resolution within 7 days, range [0, 1]"
    )
    confidence_band: str = Field(
        ..., description="low | medium | high — discrete signal for agent decision-making"
    )
    top_features: list[str] = Field(
        ..., description="Top 3 features that drove this prediction"
    )
    scope_caveat: Optional[str] = Field(
        None, description="Warning when prediction reliability is reduced (e.g. OOD repo)"
    )
    model_version: str = Field(
        ..., description="Model version string, e.g. 'v0.1-capstone'"
    )


class BatchPredictionOutput(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionOutput]
    total: int = Field(..., description="Number of issues processed")