# Issue Triage Agent

Predicting which GitHub issues get resolved within 168 hours, across major
Python open-source repositories. Phase 1 builds the classifier; Phase 2
extends it into an agent that uses the classifier as a callable tool
alongside RAG retrieval and web search.

> **Status:** Phase 1 in progress. Ingest pipeline complete. Cleaning,
> feature engineering, and training are next.

## Problem

See [`PROBLEM.md`](PROBLEM.md) for the locked problem statement,
target definition, evaluation strategy, and out-of-scope items.

## Design decisions

See [`DECISIONS.md`](DECISIONS.md) for the running ADR log. Every
non-trivial design choice is recorded with its rationale and trade-offs.

## Layout

\`\`\`
.
├── src/                  # Pipeline scripts, run in numbered order
│   ├── 01_ingest.py      # Scrape closed issues from GitHub API
│   ├── 02_clean.py       # (next) Apply 168h filter, drop bots, dedupe
│   ├── 03_features.py    # (later) Feature engineering
│   └── 04_train.py       # (later) XGBoost training + eval
├── data/
│   ├── raw/              # Raw JSON per repo (gitignored)
│   └── processed/        # cleaned.parquet, features.parquet (gitignored)
├── models/               # Trained model artifacts (gitignored)
├── PROBLEM.md
├── DECISIONS.md
└── MODEL_CARD.md         # (Day 5) Honest reporting of model behavior
\`\`\`

## Setup

\`\`\`bash
git clone git@github.com:pugal-mugilan/issue-triage-agent.git
cd issue-triage-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your GitHub PAT
\`\`\`

Generate a PAT at https://github.com/settings/tokens with no scopes
(public repo data only). Set expiration to 90 days.

## Running the pipeline

\`\`\`bash
# scrape ~30K closed issues across 5 repos (~10 minutes)
python src/01_ingest.py

# (next) clean and apply observability filter
python src/02_clean.py
\`\`\`

## Data sources

| Repo | Role |
|---|---|
| `huggingface/transformers` | Train |
| `pytorch/pytorch` | Train |
| `langchain-ai/langchain` | Train |
| `fastapi/fastapi` | Held-out (OOD eval) |
| `scikit-learn/scikit-learn` | Held-out (OOD eval) |

Train/OOD split is by repo, not by time within a repo. This lets us
measure how well the model generalizes to libraries it has never seen,
which matters more for the Phase 2 agent than within-repo accuracy.
