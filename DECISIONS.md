# Decision Log

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

---

<!-- DL-007 and DL-008: if these exist in your local repo, keep them here. -->

---

## DL-009: Drop `comments` from the feature set

**Date:** 2026-06-25
**Status:** Accepted

**Context.** The cleaned dataset includes a `comments` column — the total count of comments on each issue at the time the GitHub API was scraped. During exploratory analysis it correlates strongly with the target (issues that close as completed within 168h tend to have had a conversation first), making it tempting to include as a feature.

However, the model is designed to be called by a Phase 2 LLM agent on brand-new issues — issues at or near `created_at`, before any comments have been posted. In production, the agent's input will always have `comments = 0`. This is a classic post-filing-accumulation feature: the value in the training data reflects activity that happened *after* the prediction point would have occurred in production. Training on it teaches the model to lean on a signal that is structurally absent at inference time.

This failure mode is exactly what the time-travel test is designed to catch: "Could this value have been known at `issue.created_at`, in production, without time travel?" `comments` fails the test.

**Decision.** Drop `comments` entirely from the v0.1 feature set. Do not impute it, do not include any feature derived from it (e.g. `has_comments`).

**Alternatives considered:**

- *Include `comments` as-is.* **Rejected:** structural train/serve mismatch. The model would learn to rely on comment counts that are always zero in production, producing confident but meaningless predictions.
- *Engineer a `comments_within_first_hour` feature via a fixed post-creation delay.* Deferred to v0.2. This value *would* pass the time-travel test (it can be computed in production by waiting one hour before predicting), but it changes the operational contract from immediate prediction to delayed prediction. Out of scope for v0.1.

**Rationale.** The time-travel test is a hard gate, not a guideline. v0.1 prefers an honest model with lower metrics over a leaky model that reports artificially high F1. An F1 built on a feature the model can never see in production is not a real F1.

**Consequences:**
- Training metrics will be lower than they would have been with the leak included — this is the intended outcome.
- Production input schema is simpler: the Phase 2 agent does not need to fetch comments before calling the tool.
- v0.2 may revisit with a `comments_within_first_hour` feature if the one-hour delay is acceptable to downstream consumers.

---

## DL-010: Defer label-derived features (`labels`, `n_labels`) to v0.2

**Date:** 2026-06-25
**Status:** Accepted

**Context.** The cleaned dataset includes `labels` (list of GitHub labels applied to the issue) and `n_labels` (count of labels). These look attractive as features — `bug` vs `enhancement` could signal different resolution timelines, and label count might proxy for issue complexity.

However, in popular Python repos, most labels are maintainer-applied *during triage*, not at filing time. The scrape captures only the final label state, which reflects decisions made hours or days after `created_at`. This is the same post-filing-accumulation pattern as `comments` (DL-009): the training data encodes information the model would never have access to in production.

Restricting to labels applied within the first hour after creation would solve the leakage problem, but requires hitting GitHub's events endpoint for every issue — roughly doubling the scrape API budget per issue.

**Decision.** Drop both `labels` and `n_labels` from v0.1. Plan a v0.2 feature set that includes `first_hour_label_count` plus one-hot flags for a small fixed vocabulary (`bug`, `enhancement`, `question`, `documentation`), sourced from the events endpoint.

**Alternatives considered:**

- *Include `labels` and `n_labels` as-is.* **Rejected:** same failure family as DL-009 — maintainer triage leakage. The model would learn patterns from labels it can never see in production.
- *Implement first-hour label filtering in v0.1.* **Rejected:** doubles the scrape API budget per issue and adds significant Day 2 engineering complexity. Deferred to v0.2 where the scrape infrastructure can be upgraded properly.

**Rationale.** v0.1 prioritises a leak-free baseline over maximum feature coverage. The missing label signal is a known limitation, acknowledged explicitly in `MODEL_CARD.md`. It's better to ship a clean model that can be improved than a leaky model that must be debugged.

**Consequences:**
- Loses potentially strong signal in v0.1 (label vocabulary could meaningfully separate bug-fix quick-wins from long-running feature requests).
- `01_ingest.py` unchanged for v0.1 — no events endpoint needed.
- v0.2 scrape budget roughly doubles per issue to support first-hour label retrieval.

---

## DL-011: Drop maintainer-applied metadata and target-arithmetic columns

**Date:** 2026-06-25
**Status:** Accepted

**Context.** Day 3's column-by-column audit of `cleaned_train.parquet` surfaced four additional columns that fail the time-travel test, beyond `comments` (DL-009) and labels (DL-010):

- **`n_assignees` and `has_milestone`** — both are maintainer-applied during triage, not at filing. A newly-filed issue has zero assignees and no milestone. The training data reflects the final state after triage decisions, creating the same post-filing leakage pattern.
- **`time_to_close_hours`** — a Day 2 cleaning intermediate calculated as `(closed_at - created_at) / 3600`. This is literal target arithmetic: the column directly encodes how long the issue took to close, which is exactly what the model is trying to predict. Including it would yield ~100% F1 by handing the model the answer. This was the most dangerous column in the dataset — it survived the cleaning pipeline and would have produced a "perfect" model that does nothing useful in production.
- **`user_type`** — constant `"User"` across all rows after Day 2's bot filter removed non-human accounts. A constant column carries zero signal.

**Decision.** Add all four columns to `DROP_AS_LEAKAGE` in `src/03_features.py`, ensuring they are explicitly excluded before any model sees the data. Additionally, a TODO was logged for v0.2: drop `time_to_close_hours` at the end of `02_clean.py` itself, so that no downstream script can accidentally inherit the leak.

**Alternatives considered:**

- *Trust the cleaning pipeline to handle `time_to_close_hours`; don't add an explicit drop in `03_features.py`.* **Rejected:** defense in depth. If a future script reads the cleaned parquet directly, it inherits the leak. The drop must live in code, not in operator memory.
- *Engineer first-hour `n_assignees` or `has_milestone` features.* Deferred to v0.2, alongside first-hour label features (DL-010). The same scrape infrastructure upgrade unlocks all three.

**Rationale.** Every column in the dataset must receive an explicit verdict before training: keep, drop, or defer. The `time_to_close_hours` near-miss demonstrated why this discipline matters — a column that looks like a reasonable feature was actually the target in disguise. The audit discipline must live in code (the `DROP_AS_LEAKAGE` list), not in operator habit.

**Consequences:**
- `DROP_AS_LEAKAGE` in `03_features.py` now covers 11 columns total. The full list lives inline in the source code as documentation.
- v0.2 TODO: `02_clean.py` will strip `time_to_close_hours` immediately after target derivation, preventing the leak from propagating.
- Any new repos added to the training set in the future must undergo the same column-by-column audit before inclusion.

---

## DL-012: Model selection and per-repo acceptance gate

**Date:** 2026-07-05
**Status:** Accepted

**Context.** Day 4 trained five model variants against a three-tier baseline ladder (majority-class brick, stratified random coin, logistic regression with `class_weight="balanced"`). Key results on the held-out test set (n=2,052), positive-class F1:

| Model                | Overall | transformers | langchain | pytorch |
|----------------------|---------|-------------|-----------|---------|
| Brick (majority)     | 0.0000  | 0.0000      | 0.0000    | 0.0000  |
| Coin (stratified)    | 0.3737  | 0.4083      | 0.2857    | 0.3758  |
| Logistic Regression  | 0.5122  | 0.5913      | 0.0000    | 0.5032  |
| XGBoost              | 0.5200  | 0.5846      | 0.3103    | 0.5117  |
| NN (unweighted)      | 0.5500  | 0.6082      | 0.2444    | 0.5450  |
| NN (repo-weighted)   | 0.5432  | 0.5887      | 0.4154    | 0.5366  |

Findings:

1. **Logistic regression collapsed to the brick** until `class_weight="balanced"` was added. With 63% negative class, the default loss function found that predicting all-NO minimised total loss. This served as a canary: XGBoost (`scale_pos_weight`) and the NN (`pos_weight`) would face the same collapse without explicit class-imbalance handling.
2. **XGBoost barely beat LR overall** (+0.008 F1) and failed the transformers per-repo gate consistently (1/5 seeds passed in `05a_seed_check.py`). The added complexity of 101 boosted trees bought almost nothing over a 19-parameter linear model.
3. **The unweighted NN broke the ~0.51 ceiling** (F1=0.55), suggesting the features carry non-linear signal that trees didn't capture. However, langchain fell below random chance (0.24 < coin's 0.29), making it unusable for that repo.
4. **Repo-weighted NN raised langchain from 0.24 to 0.42** by assigning each training sample a weight inversely proportional to its repo's frequency. This came at a cost of 0.007 overall F1 — a tradeoff that produces the highest floor across all repos, with no catastrophic slice.

**Decision.** Select the repo-weighted 2-layer NN (`06b_nn_weighted.py`) as the v0.1 production model.

Acceptance gate (revised from the original strict per-repo proposal): a model is accepted when it (a) beats the stratified-random baseline on overall positive-class F1, AND (b) no single in-domain repo has F1 = 0.00. The gate tests for *catastrophic* slice failure, not strict per-repo superiority over every other model — because enforcing strict superiority forces tradeoffs that degrade the weakest slice.

**Alternatives considered:**

- *XGBoost as primary model.* **Rejected:** barely beat LR overall; failed the transformers gate in 4/5 seeds; the added complexity of gradient-boosted trees was not justified given that the NN outperformed it.
- *Unweighted NN as primary model.* **Rejected:** langchain F1 (0.24) was below random chance. The agent consuming this tool would get worse-than-random predictions on langchain issues, eroding trust across all repos.
- *Continue tuning XGBoost hyperparameters.* **Deferred:** the NN already exceeded XGBoost's best result without any tuning of either model. If the NN proves unstable in OOD evaluation, XGBoost tuning is the fallback path.
- *Strict per-repo gate (each repo must beat LR's per-repo F1).* **Rejected:** this gate forced models into tradeoffs where improving one repo required degrading another. The revised gate checks for catastrophic failure (F1 = 0.00) rather than strict superiority, which better reflects the agent's needs.

**Rationale.** For a multi-repo agent tool, the worst-repo F1 matters more than the best-repo F1. A tool that returns garbage on one repo erodes agent trust for all repos, even the ones where the model performs well. The repo-weighted NN has the highest floor (langchain=0.42) while maintaining competitive overall F1 (0.54). The unweighted NN has a higher ceiling on its best repos, but one repo is essentially useless — and a tool with a useless mode is worse than a tool with a mediocre mode.

**Consequences:**
- `models/nn_weighted_model.pt` is the v0.1 production artifact. `models/xgb_model.json` is retained as a backup in case OOD evaluation reveals NN instability.
- `MODEL_CARD.md` must document langchain as the weakest slice (F1=0.42, highest variance across seeds) and recommend lower-confidence treatment for langchain predictions.
- OOD evaluation on held-out repos (fastapi, scikit-learn) is the next validation step. If the NN fails OOD, this decision will be revisited.
- v0.2 improvement path: collect more langchain training data, and/or engineer langchain-specific features to close the per-repo gap without sample weighting.

---

## DL-013: Precision@5 miss — feature ceiling, not model ceiling

**Date:** 2026-07-05
**Status:** Accepted

**Context.** PROBLEM.md set a success criterion of in-domain P@5 ≥ 0.70. Day 5 evaluation (`07_precision_at_k.py`) measured P@5 = 0.60 overall (3/5 correct). Per-repo results: transformers 0.60, langchain 0.40, pytorch 0.60. P@10 overall = 0.40, P@20 = 0.50.

Probability distribution diagnostic revealed the root cause: the model's top-ranked predictions cluster between 0.71 and 0.80, with the top 10 issues mixing correct and incorrect predictions at nearly identical confidence levels (e.g., #1 = 0.8014 ✅, #2 = 0.7993 ❌, #3 = 0.7546 ❌). The model cannot separate true positives from false positives at the top of the ranking.

The model's overall probability spread is narrow: the middle 50% of predictions fall between 0.47 and 0.56 (a 0.09 range), with standard deviation 0.13. The model distinguishes "roughly positive" from "roughly negative" (F1=0.54) but lacks the signal resolution to rank confidently within the positive group.

**Decision.** Accept the P@5 miss as a known limitation of v0.1. Document it in MODEL_CARD.md. Defer the fix to v0.2, where text-based features (TF-IDF on title/body, and later embeddings) will provide the discriminative signal the current 18 structural features lack.

**Alternatives considered:**

- *Sprint to add TF-IDF features on Day 5.* **Rejected:** text feature engineering (tokenization, vocabulary fitting, dimensionality reduction) is a significant scope addition that risks breaking the existing leak-free pipeline under time pressure. Text features are the core topic of Week 13 (Phase 2) and belong there.
- *Lower the P@5 threshold to 0.50.* **Rejected:** retroactively adjusting a success criterion to match results is dishonest. The 0.70 threshold was set based on the product question ("would a maintainer trust the top 5?"), and the answer at 0.60 is still "not reliably."
- *Switch to XGBoost (which might rank differently).* **Rejected:** XGBoost's overall F1 was lower (0.52 vs 0.54), and both models share the same 18 features. The ranking ceiling is a feature problem, not a model problem — switching models won't fix it.

**Rationale.** The 18 structural features (title length, body length, has code block, day of week, author association, etc.) carry enough signal for binary classification but not for fine-grained ranking at the top. Two issues with similar structural profiles — both have code blocks, similar body lengths, filed midweek by contributors — are indistinguishable to the model even if one is a docstring typo (quick fix) and the other is an architecture redesign (not quick). Text content is the missing discriminative signal.

**Consequences:**
- MODEL_CARD.md must list P@5 = 0.60 as a known limitation, with explicit guidance: "Do not use the model's top-K ranking as a sole triage criterion in v0.1. Use the binary prediction (above/below 0.50) as a filter, then apply human judgment within the filtered set."
- v0.2 improvement path: add TF-IDF features on issue title and body (Week 13), then sentence embeddings (Week 15). Re-evaluate P@5 after each addition.
- The agent integration contract (Week 14) should expose `confidence_band` categories (low/medium/high) rather than raw probability ranking, since the probabilities at the top are not well-separated enough for ordinal ranking.