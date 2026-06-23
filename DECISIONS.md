 Decision Log
 
This file records design decisions made during the GitHub Issue Triage capstone. Each decision is dated, numbered, and includes the alternatives considered along with reasons for rejection.
 
Use this file to understand *why* the project looks the way it does. Update it whenever a meaningful design choice is made — even a small one.
 
**Format.** Lightweight ADR (Architecture Decision Record). Each entry has: Context, Decision, Alternatives considered (with rejection reasons), Rationale, Consequences.
 
---
 
## DL-001: Real GitHub data instead of demo datasets
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** Initial framing offered three well-known tutorial datasets: Telco Customer Churn, NIFTY-50 stock movement, Diabetes 130-US Hospitals. All are standard bootcamp / Kaggle / UCI datasets.
 
**Decision.** Use live data from GitHub's REST and GraphQL APIs across multiple Python OSS repositories.
 
**Alternatives considered:**
 
- *Telco Customer Churn.* Clean tabular data, strong XGBoost story, realistic class imbalance. **Rejected:** demo dataset; signals "ML 101 portfolio" rather than real engineering. Every bootcamp grad ships this.
- *NIFTY-50 stock movement.* Owner's earlier domain; existing deployed classifier from Week 7. **Rejected:** stock direction prediction is fundamentally hard (even good models barely clear 55%). Capstone would *look* weak despite strong engineering. Deferred to Week 14 Phase 2 agent demo as a second tool the agent can call.
- *Diabetes 130-US Hospitals.* Real-world healthcare impact narrative. **Rejected:** dataset is messy in ways that consume Day 2 time. Sensitive domain demands deeper Day 5 calibration work.
**Rationale.** Real OSS data has live API quirks, current data, and maps to a recognized product category (CodeRabbit, Sweep AI, Linear AI Triage, GitHub auto-triage). Aligns with the owner's 5-year software dev background — the hiring story tells itself.
 
**Consequences:**
- Day 2 includes a GitHub API scraper (~2 hours)
- Free GitHub personal access token required (5,000 req/hr rate limit)
- Stronger interview narrative; recognized product space
- Forces honest engineering on messy, dynamic data — not a pre-cleaned CSV
---
 
## DL-002: Multi-repo training with held-out OOD repos
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** First sketch trained on `huggingface/transformers` alone, with a model card disclaiming single-repo scope.
 
**Decision.** Train on 3 in-domain Python ML/dev-tool repos: `huggingface/transformers`, `pytorch/pytorch`, `langchain-ai/langchain`. Hold out 2 OOD repos (`fastapi/fastapi`, `scikit-learn/scikit-learn`) entirely from training for honest generalization evaluation.
 
**Alternatives considered:**
 
- *Single repo (`huggingface/transformers` only).* **Rejected:** scope too narrow, weak generalization story, use cases collapse to "only useful for ML library maintainers."
- *All 5 repos in training, internal train/test split.* **Rejected:** no honest OOD evaluation possible. The model card cannot credibly claim generalization without held-out data.
- *Per-customer fine-tuning at training time.* **Rejected:** too complex for a 2-week capstone. Deferred to Week 15 LoRA exercise where it fits naturally.
**Rationale.** Held-out OOD repos enable defensible generalization claims for the model card. Three training repos give domain diversity (ML library, deep learning framework, LLM tooling) without overwhelming the scraper. Combined volume (~33K closed issues) is more than enough for XGBoost.
 
**Consequences:**
- Scraper runs on 5 repos instead of 1
- Model card reports in-domain AND OOD metrics as separate rows
- Official scope claim becomes "Python ML/dev-tool OSS ecosystem"
- ~33K training issues expected; storage and compute remain trivial
---
 
## DL-003: Target = "closed-as-completed by human within 168h"
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** Initial framing was "closed within 7 days." Vague; would conflate distinct phenomena into one label.
 
**Decision.** A positive example requires all four:
 
1. `state == "closed"`
2. `state_reason == "completed"`
3. Closing actor is a human (excludes bot accounts such as `stale-bot`, `dependabot[bot]`)
4. All occurred within 168 hours of `created_at`
**Alternatives considered:**
 
- *"Closed within 7 days" with no other filters.* **Rejected:** conflates real resolution with abandonment-by-`stale-bot` and `wontfix` closures. Three different phenomena with three different feature signatures.
- *Include `state_reason == "duplicate"` as positive.* **Rejected:** duplicates are closed by linking elsewhere, not by resolving the underlying issue. Different decision pattern.
- *14-day window.* **Rejected:** blurs "quick resolution" with "moderate-complexity work" and weakens signal.
**Rationale.** The four conditions encode "real resolution." A looser definition trains a model to predict a noisy mix of *resolved + abandoned + duplicate*, which collapses three distinct phenomena into one fuzzy target.
 
**Consequences:**
- Scraper (Day 2) must filter on `state_reason` and bot accounts
- Bot account allowlist required (start with known names; expand as encountered)
- Training signal is cleaner; reduces label noise
---
 
## DL-004: Prediction horizon = 168 hours
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** Need a precise horizon that matches the maintainer's decision cadence and supports a clean training pipeline.
 
**Decision.** Predict whether resolution (per DL-003) occurs within 168 hours of `issue.created_at`.
 
**Alternatives considered:**
 
- *24 hours.* **Rejected:** too narrow; captures only typo-level fixes. Trivial problem.
- *72 hours (3 days).* Considered. Reasonable, but doesn't match the maintainer's weekly mental model.
- *14 days.* **Rejected:** blurs quick wins with moderate work; weakens the signal that separates them.
- *30 days.* **Rejected:** asks "resolved ever" rather than "quick wins" — wrong question entirely.
**Rationale.** 168 hours matches the typical OSS maintainer's weekly triage rhythm. Maintainers work their backlog Monday morning and ask *"what's likely done by next Monday?"* The horizon must match that question. 7-day MTTR is also the industry-standard OSS health metric.
 
**Consequences:**
- Any issue scraped less than 168h ago has no observable label → filtered from training
- API contract advertises 168h, not "7 days" (eliminates timezone ambiguity)
- Class balance estimable only on older issues
---
 
## DL-005: Precision@5 and precision@10 as primary metrics
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** Standard ML capstones default to F1 or accuracy as the headline metric for model selection.
 
**Decision.** Day 5 model selection driven by **precision@5** and **precision@10**. F1, precision, recall reported for completeness but not used for ranking models against each other.
 
**Alternatives considered:**
 
- *F1 as primary.* **Rejected:** A model with higher F1 can have terrible precision@5 if false positives cluster at the top of the ranking. F1 does not reflect the user's actual experience.
- *Accuracy as primary.* **Rejected:** useless under class imbalance.
- *NDCG (Normalized Discounted Cumulative Gain).* Considered. More sophisticated ranking metric, but precision@K is closer to the user's actual interaction pattern (they look at top 5, not at full ranking).
**Rationale.** The downstream user (maintainer) only ever sees the top 5–10 ranked issues. The metric that mirrors this is precision@K. Selecting models on F1 alone would risk shipping a model that scores well on paper but ranks badly in production.
 
**Consequences:**
- Evaluation code computes precision@K (one ranking step per evaluation)
- Model card reports precision@5 prominently
- Day 4 baselines report precision@K as a sanity row (even though majority-class baseline doesn't rank meaningfully)
---
 
## DL-006: Agent-facing API contract with structured outputs
 
**Date:** 2026-06-16
**Status:** Accepted
 
**Context.** The model is the first tool that an LLM agent (Week 14+) will call. The API must serve a machine consumer, not just a human dashboard.
 
**Decision.** API returns structured Pydantic objects per prediction:
 
- `p_close_7d: float`
- `confidence_band: Literal["low", "medium", "high"]`
- `top_features: list[str]`
- `scope_caveat: str | None`
**Alternatives considered:**
 
- *Numeric output only (`p_close_7d`).* **Rejected:** agent cannot reason about uncertainty or scope without extra fields.
- *Free-text explanation field.* **Rejected:** harder to evaluate, less reliable for downstream reasoning, prone to hallucination if generated.
- *Return only top-K results.* **Rejected:** the agent may want to reason about lower-ranked items; filtering should happen on the agent side, not in the tool.
**Rationale.** A human dashboard tolerates ambiguous outputs (charts, free text). An LLM agent needs structured, typed, machine-readable output. `confidence_band` lets the agent communicate uncertainty without doing probability math. `top_features` enables agent-generated reasoning ("flagged because: short body, experienced author, has stack trace"). `scope_caveat` lets the agent refuse gracefully on OOD repos.
 
**Consequences:**
- Day 6 FastAPI app uses these Pydantic models directly as request/response shapes
- Per-prediction feature importance must be computable (XGBoost built-in importance suffices for Week 11–12; SHAP added in Week 26)
- Calibration step required on Day 5 to produce honest `confidence_band` thresholds
- Output schema is stable; Week 14 agent can be built against it without changes
 