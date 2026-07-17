# Model Card — GitHub Issue Triage Classifier v0.1

**Model type:** Binary classifier (2-layer neural network with repo-based sample weighting)
**Task:** Predict whether a newly-opened GitHub issue will be closed-as-completed within 168 hours (7 days)
**Owner:** Pugal Mugilan
**Date:** 2026-07-05
**Status:** v0.1 — Phase 1 Capstone

---

## Intended use

This model is designed to be called as a **tool by an LLM agent** acting on behalf of an OSS maintainer. The agent passes an issue's metadata at filing time, and the model returns a resolution probability, confidence band, and top contributing features.

**Primary use case:** Filter newly-opened issues into "likely quick resolution" vs "likely not," helping maintainers prioritize limited triage time.

**Not intended for:**
- Sole triage criterion (v0.1 ranking at the top is not reliable enough — see Known Limitations)
- Non-Python repositories
- Repos outside the ML library / dev-tool / web framework / data library domain
- Replacing human judgment on individual issues

---

## Architecture

| Component | Detail |
|---|---|
| Type | Feedforward neural network |
| Layers | 18 → 64 (ReLU, Dropout 0.3) → 32 (ReLU, Dropout 0.3) → 1 (Sigmoid) |
| Parameters | 3,329 |
| Training | Adam (lr=0.001), BCEWithLogitsLoss with pos_weight + per-sample repo weights |
| Early stopping | Patience 10 epochs on validation loss |
| Framework | PyTorch |

**Sample weighting:** Each training row is weighted inversely proportional to its repo's frequency, so underrepresented repos (langchain) contribute equally to loss despite having fewer rows. Combined with pos_weight for class imbalance.

---

## Training data

| Attribute | Value |
|---|---|
| Source | GitHub REST API (issues endpoint, paginated) |
| Repos (in-domain) | `pytorch/pytorch`, `huggingface/transformers`, `langchain-ai/langchain` |
| Date range | Issues created within each repo's active period, filtered to those with ≥168h observability |
| Raw scrape | 68,816 items across 5 repos |
| After cleaning | 10,260 rows (7-stage funnel: drop PRs → date filter → observability → drop bots → drop null state_reason → dedupe → label) |
| Class balance | 36.9% positive (resolved within 168h), 63.1% negative |
| Train/val/test split | 6,566 / 1,642 / 2,052 (stratified, seed=42) |

**Repo distribution in training set:**

| Repo | Rows | Share |
|---|---|---|
| pytorch/pytorch | ~6,500 | 63% |
| huggingface/transformers | ~2,300 | 22% |
| langchain-ai/langchain | ~1,460 | 14% |

pytorch dominates. Sample weighting compensates during training, but the underlying data imbalance means langchain estimates have higher variance.

---

## Features (18 total)

All features pass the time-travel test: they are knowable at `issue.created_at` without future information.

**Text features (6):** title_length, body_length, body_word_count, body_has_code_block, body_has_stacktrace, body_has_url

**Time features (4):** hour_of_day, day_of_week, is_weekend, month

**One-hot — repo (3):** repo_huggingface/transformers, repo_langchain-ai/langchain, repo_pytorch/pytorch

**One-hot — author association (4):** aa_COLLABORATOR, aa_CONTRIBUTOR, aa_MEMBER, aa_NONE

**Aggregation (1):** author_prior_count (number of prior issues by the same author in training data, computed via merge_asof to prevent leakage)

**Features explicitly dropped (with rationale):**
- `time_to_close_hours` — literal target arithmetic (closed_at − created_at). Would yield 100% F1 and a useless production model. Near-miss: survived initial cleaning, caught by schema inspection before training.
- `comments` — post-filing accumulation (ADR-009)
- `labels`, `n_labels` — post-filing maintainer triage (ADR-010)
- `n_assignees`, `has_milestone` — post-filing maintainer triage (ADR-011)
- `user_type` — constant after bot filter; no signal

---

## Performance

### In-domain (test set, n=2,052)

| Model | Overall F1 | transformers | langchain | pytorch |
|---|---|---|---|---|
| Brick (majority class) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Coin (stratified random) | 0.3737 | 0.4083 | 0.2857 | 0.3758 |
| Logistic Regression | 0.5122 | 0.5913 | 0.0000 | 0.5032 |
| XGBoost | 0.5200 | 0.5846 | 0.3103 | 0.5117 |
| NN (unweighted) | 0.5500 | 0.6082 | 0.2444 | 0.5450 |
| **NN (repo-weighted) ← v0.1** | **0.5432** | **0.5887** | **0.4154** | **0.5366** |

### In-domain Precision@K

| K | P@K | Correct/K |
|---|---|---|
| 5 | 0.6000 | 3/5 |
| 10 | 0.4000 | 4/10 |
| 20 | 0.5000 | 10/20 |

**P@5 = 0.60 — below the 0.70 target set in PROBLEM.md.** See Known Limitations and DL-013 for root cause analysis.

### Out-of-domain (held-out repos, n=680)

| Repo | Rows | Positive rate | F1 | P@5 | P@10 |
|---|---|---|---|---|---|
| fastapi/fastapi | 82 | 73.2% | 0.5688 | 0.8000 | 0.7000 |
| scikit-learn/scikit-learn | 598 | 47.0% | 0.5825 | 0.8000 | 0.8000 |
| **Combined** | **680** | **50.1%** | **0.5806** | **0.8000** | **0.8000** |

**OOD P@5 = 0.80 — passes the 0.55 threshold.** However, this result should be interpreted with caution: fastapi has a 73.2% positive base rate (most issues resolve quickly), which inflates ranking metrics. The model generalizes at the classification level (F1 comparable to in-domain) but the high P@K partly reflects easier target distributions, not superior ranking ability.

---

## Success criteria status

| Criterion | Target | Actual | Status |
|---|---|---|---|
| In-domain F1 ≥ baseline + 0.15 | ≥ 0.3737 + 0.15 = 0.5237 | 0.5432 | ✅ Pass |
| In-domain P@5 | ≥ 0.70 | 0.60 | ❌ Fail |
| OOD P@5 | ≥ 0.55 | 0.80 | ✅ Pass |
| API latency ≤ 500ms for 50 issues | — | Not yet measured (Week 12) | ⏳ Pending |
| Model card published | — | This document | ✅ Pass |

---

## Known limitations

**1. Ranking at the top is unreliable (P@5 = 0.60).** The 18 structural features create a feature ceiling. The model can classify (F1=0.54) but cannot confidently rank which positive issues are *most* likely to resolve. At the top of the ranking, probabilities cluster between 0.71 and 0.80, mixing true and false positives at nearly identical confidence. Root cause: two issues with similar structural profiles (both have code blocks, similar body lengths, filed midweek by a contributor) are indistinguishable without text content. **Guidance: do not use top-K ranking as a sole triage criterion in v0.1. Use the binary prediction (above/below 0.50) as a filter, then apply human judgment within the filtered set.**

**2. Langchain is the weakest slice (F1=0.42).** Langchain has the fewest training rows (~1,460) and highest variance across random seeds. Sample weighting lifted it from 0.24 (below random chance) to 0.42, but it remains the least reliable repo. Predictions on langchain-style repos (small, fast-moving, agent/LLM tooling) should be treated with lower confidence.

**3. No text understanding.** The model has no access to what the issue actually says — only structural signals (length, patterns like code blocks or URLs). An issue titled "typo in docstring" and one titled "redesign the tokenizer architecture" look similar if they have similar body lengths and code blocks. Text features (TF-IDF, embeddings) are the planned fix for v0.2.

**4. OOD P@5 is inflated by high base rates.** Fastapi's 73.2% positive rate means even a random ranker would score well. The OOD P@5 = 0.80 result should not be interpreted as strong generalization — it reflects an easier target distribution. Repos with lower positive rates (like langchain at 25.5%) are a harder and more honest test.

**5. Repo one-hot features are useless for unseen repos.** For repos not in the training set, all three repo one-hot columns are zero. The model falls back on the remaining 15 features. This means predictions for new repos carry less information and higher uncertainty.

**6. Probability calibration not formally tested.** The model outputs probabilities via sigmoid, but whether "0.70 probability" actually means "70% of such issues resolve" has not been verified with a calibration curve. Probabilities should be treated as relative rankings, not absolute likelihoods.

---

## Interface (preview — Week 12 implementation)

**Input schema:**
```python
class IssueInput(BaseModel):
    title: str
    body: str
    repo: str                          # e.g. "pytorch/pytorch"
    author_association: str            # COLLABORATOR | CONTRIBUTOR | MEMBER | NONE
    user_login: str
    created_at: datetime               # ISO 8601, UTC
```

**Output schema:**
```python
class PredictionResponse(BaseModel):
    p_close_7d: float                  # probability in [0, 1]
    prediction: bool                   # True if p_close_7d >= 0.50
    confidence_band: str               # "low" | "medium" | "high"
    top_features: list[str]            # top 3 contributing features
    scope_caveat: str | None           # warning if repo is OOD or weak slice
```

**Confidence bands:**
- **high:** p_close_7d ≥ 0.65 or p_close_7d ≤ 0.35
- **medium:** 0.50–0.65 or 0.35–0.50
- **low:** 0.45–0.55 (model is effectively uncertain)

**Scope caveats (examples):**
- Repo not in training set → "This repo was not in the training data. Prediction reliability is reduced."
- Langchain or similar small repo → "This repo is in a weak-performance slice. Treat prediction with lower confidence."

---

## Versioning

| Version | Date | Changes | Test F1 | P@5 |
|---|---|---|---|---|
| **v0.1** | 2026-07-05 | Initial release. 18 structural features, repo-weighted NN. | 0.5432 | 0.60 |
| v0.2 (planned) | Week 13 | Add TF-IDF features on title and body | — | Target ≥ 0.70 |
| v0.3 (planned) | Week 15 | Add sentence embeddings, per-repo LoRA adapters | — | — |

---

## Reproduction

```bash
# From repo root
python src/01_ingest.py          # scrape GitHub API (requires PAT in .env)
python src/02_clean.py           # 7-stage cleaning → cleaned_train.parquet
python src/03_features.py        # feature engineering → X_train/X_test + artifacts
python src/06b_nn_weighted.py    # train repo-weighted NN → nn_weighted_model.pt
python src/07_precision_at_k.py  # in-domain P@K evaluation
python src/08_ood_eval.py        # OOD evaluation on fastapi + scikit-learn
```

Dependencies pinned in `requirements.txt`. Random seed = 42 for all splits and training.

---

## References

- PROBLEM.md — problem framing and success criteria
- DECISIONS.md — full decision log (DL-001 through DL-013)
- DATANOTES.md — cleaning decisions and data audit

## Latency (Containerized, CPU-only)

| Metric | Value |
|---|---|
| Batch 50 issues (p50) | 123 ms |
| Batch 50 issues (max) | 133 ms |
| Single issue (p50) | 6.0 ms |
| Single issue (p95) | 7.0 ms |
| Single issue (p99) | 7.7 ms |

Measured via `benchmark_latency.py` against Dockerized service (`python:3.14-slim`, CPU-only PyTorch).
PROBLEM.md target: ≤ 500 ms for 50 issues → **PASS** (123 ms, 4× under budget).