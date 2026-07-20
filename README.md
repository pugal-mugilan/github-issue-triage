# GitHub Issue Triage

Predicts whether a newly opened GitHub issue will be resolved within 7 days, trained on real data from `pytorch/pytorch`, `langchain-ai/langchain`, and `huggingface/transformers`.

Built as a Phase 1 capstone — an end-to-end ML system designed from day one to be called as a tool by an LLM agent in Phase 2.

## Why This Exists

OSS maintainers spend 30–60 minutes per triage session deciding which issues to look at first. This classifier estimates the probability that a new issue will be closed-as-completed within 168 hours (7 days), so maintainers — or an LLM agent acting on their behalf — can prioritize quick-resolution issues during limited triage windows.

Issue triage is a recognized product category (CodeRabbit, Sweep AI, Linear's AI triage, GitHub's built-in auto-triage). This project tackles the same problem with a lightweight structural classifier, with text features planned for v0.2.

## Results

| Criterion | Target | Actual | Status |
|---|---|---|---|
| In-domain F1 ≥ baseline + 0.15 | ≥ 0.5237 | 0.5432 | ✅ Pass |
| In-domain Precision@5 | ≥ 0.70 | 0.60 | ❌ Fail (documented) |
| OOD Precision@5 | ≥ 0.55 | 0.80 | ✅ Pass (caveat below) |
| API latency ≤ 500ms (50 issues) | p50 ≤ 500ms | p50 = 123ms | ✅ Pass |
| Model card published | — | MODEL_CARD.md | ✅ Pass |

**In-domain P@5 missed its threshold.** The 18 structural features lack the discriminative signal to reliably rank the top 5 issues. Two issues with similar body lengths, code blocks, and filing times are indistinguishable to the model — it classifies (positive vs negative) but can't rank within the positive group. This is a feature ceiling, not a model ceiling. Text features (TF-IDF, embeddings) are the fix path for v0.2.

**OOD P@5 passed, but with a caveat.** The held-out repos (especially `fastapi` at 73.2% positive base rate) have easier target distributions. High OOD P@K partly reflects that, not superior generalization.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training Pipeline (offline — run_pipeline.py)          │
│                                                         │
│  Raw Issues → Ingest → Clean → Features → Train NN     │
│                                         ↓               │
│                                  Saved Artifacts        │
│                                  (.pt, scaler, encoder) │
└──────────────────────────────────┬──────────────────────┘
                                   │ loaded at startup
┌──────────────────────────────────▼──────────────────────┐
│  Serving Pipeline (online — Docker + FastAPI)           │
│                                                         │
│  Request → Pydantic → Features → NN Inference → Response│
│  (JSON)    Validate   app/       torch.no_grad   (JSON) │
│                       features.py                       │
└─────────────────────────────────────────────────────────┘
```

Full diagram: [docs/architecture.mermaid](docs/architecture.mermaid)

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/pugal-mugilan/github-issue-triage.git
cd github-issue-triage
docker compose up --build
```

The service starts at `http://localhost:8000`. Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fix typo in README",
    "body": "Line 42 has a spelling error.",
    "repo": "pytorch/pytorch",
    "author_association": "CONTRIBUTOR",
    "user_login": "octocat",
    "created_at": "2025-01-15T10:30:00Z"
  }'
```

Interactive API docs at `http://localhost:8000/docs`.

### Option B: Local Python

```bash
git clone https://github.com/pugal-mugilan/github-issue-triage.git
cd github-issue-triage
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Training Pipeline (optional)

```bash
# Run the full pipeline: ingest → clean → features → train → evaluate
python run_pipeline.py

# Run tests
python -m pytest tests/ -v
```

**Note:** Ingestion (`01_ingest.py`) requires a GitHub PAT in `.env`. See `.env.example`. Without it, the pipeline runs from cached parquet files in `data/`.

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Single issue → prediction with confidence band |
| POST | `/predict/batch` | List of issues → list of predictions |
| GET | `/model/info` | Model version, features, train metrics, limitations |
| GET | `/health` | Liveness check (503 if model not loaded) |

Every prediction returns: `predicted_class`, `p_close_7d`, `confidence_band` (low/medium/high), `top_features`, `scope_caveat`, and `model_version`. See [INTEGRATION.md](INTEGRATION.md) for full schemas and agent-facing usage.

## How It Works

**Data:** 10,260 cleaned issues from 3 training repos, scraped via the GitHub REST API. 680 held-out OOD issues from `fastapi/fastapi` and `scikit-learn/scikit-learn` (never seen during training).

**Features:** 18 leak-free structural features — body length, code block count, author history, repo identity, time-of-day, author association, and more. Every feature passes the time-travel test: it must be knowable at `issue.created_at` in production, without access to future data.

**Model:** 2-layer neural network (PyTorch) with repo-weighted loss. Selected over logistic regression (F1=0.51), XGBoost (F1=0.52, failed stability gate), and an unweighted NN (F1=0.55 overall but langchain F1=0.24). Repo-weighting lifted the worst per-repo floor from 0.24 to 0.42 while maintaining overall F1=0.54.

**Baselines:** Three-tier honest comparison — majority-class predictor (F1=0.00), coin-flip (F1=0.37), logistic regression (F1=0.51). The selected model beats all three.

## Project Structure

```
github-issue-triage/
├── app/                           # Serving layer (FastAPI)
│   ├── schemas.py                 #   Pydantic request/response models
│   ├── features.py                #   Inference-time feature engineering
│   └── main.py                    #   Endpoints + model loading
├── src/                           # Training pipeline
│   ├── 01_ingest.py               #   GitHub API scraper
│   ├── 02_clean.py                #   7-stage cleaning pipeline
│   ├── 03_features.py             #   18 leak-free features
│   ├── 04_baseline.py             #   3-tier baselines
│   ├── 05_xgboost.py              #   XGBoost + stability gate
│   ├── 06_nn.py                   #   Base NN
│   ├── 06b_nn_weighted.py         #   Repo-weighted NN (selected model)
│   ├── 07_precision_at_k.py       #   P@K evaluation
│   └── 08_ood_eval.py             #   OOD generalization eval
├── tests/                         # Test suite (31 tests)
│   ├── golden/
│   │   └── golden_set.json        #   25 frozen issues with expected labels
│   ├── test_regression.py         #   Model-level regression tests
│   └── test_api.py                #   API contract tests
├── docs/
│   └── architecture.mermaid       # System architecture diagram
├── Dockerfile                     # Container definition
├── docker-compose.yml             # One-command local deployment
├── run_pipeline.py                # Single-command training orchestrator
├── test_pipeline.py               # Pipeline smoke tests (17 tests)
├── benchmark_latency.py           # Latency measurement script
├── generate_golden_set.py         # One-time golden set generator
├── data/                          # Raw + processed data (gitignored)
├── models/                        # Saved artifacts (gitignored)
├── PROBLEM.md                     # Problem framing + success criteria
├── DECISIONS.md                   # 13 ADRs (DL-001 through DL-013)
├── MODEL_CARD.md                  # Metrics, limitations, interface schema
├── INTEGRATION.md                 # Agent-facing tool contract
└── requirements.txt               # Pinned dependencies
```

## Reproducibility

Random seeds are locked across Python, NumPy, and PyTorch. The pipeline runs as a single process (no subprocesses) to guarantee seed propagation. Two consecutive runs produce identical metrics.

The regression test suite (`tests/`) catches metric drift: if F1 drops below 0.39 or the API contract changes, tests fail.

## Key Design Decisions

Every meaningful choice is documented as an ADR in [DECISIONS.md](DECISIONS.md). Highlights:

- **DL-001:** Real GitHub data instead of demo datasets — forces honest engineering on messy, dynamic data
- **DL-002:** Multi-repo training with held-out OOD repos — enables defensible generalization claims
- **DL-006:** Three-tier baseline ladder — majority-class → coin-flip → logistic regression, each must be beaten before moving to the next model
- **DL-012:** Repo-weighted NN over unweighted NN — sacrificed 1 point of overall F1 to fix a broken per-repo floor (langchain 0.24 → 0.42)
- **DL-013:** P@5 miss documented honestly — feature ceiling diagnosis, not threshold retrofit

## Known Limitations

- **No text features.** The model has no access to what the issue actually says — only structural signals. This is the root cause of the P@5 miss. Planned for v0.2.
- **Python ecosystem only.** Trained on Python ML/dev-tool/web repos. Out of scope: Rust, mobile, frontend, non-Python.
- **Small author overlap.** OOD repos have different contributor pools. Author history features are mostly zero for OOD predictions.
- **Training-serving feature gap.** `author_prior_count` is computed differently in the training pipeline vs the serving layer (serving defaults to 0 for unknown authors). Doesn't flip predictions, but probabilities differ slightly. Documented in `INTEGRATION.md`.
- **No per-repo personalization.** All repos share one model. Per-repo fine-tuning deferred to a later phase.

## What's Next

- **v0.1 (current):** Trained model, FastAPI service, Docker deployment, regression test suite, agent tool contract
- **v0.2:** Text features (TF-IDF/embeddings) to address P@5 miss, CPU-optimized Docker image
- **Phase 2 (Week 14+):** This classifier becomes the first callable tool for a Production Agentic RAG system

## Documentation

- [PROBLEM.md](PROBLEM.md) — Target definition, prediction horizon, success criteria, scope
- [DECISIONS.md](DECISIONS.md) — 13 Architecture Decision Records
- [MODEL_CARD.md](MODEL_CARD.md) — Training data, metrics, known limitations
- [INTEGRATION.md](INTEGRATION.md) — Agent-facing tool contract: endpoints, schemas, confidence bands, function-calling spec

## Tech Stack

Python · PyTorch · FastAPI · Pydantic · Docker · scikit-learn · pandas · NumPy · pytest · GitHub REST API