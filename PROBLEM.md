GitHub Issue Triage Classifier
Status: Active — Phase 1 Capstone (Weeks 11–12)
Owner: Pugal Mugilan
Last reviewed: 2026-06-16

Problem statement
OSS maintainers spend significant time deciding which newly-opened issues to triage first. This project predicts the probability that a newly-opened GitHub issue will be resolved within 7 days, enabling maintainers (or an LLM agent acting on their behalf) to prioritize quick-resolution issues during limited triage sessions.
Motivation
Issue triage is a recognized product category — CodeRabbit, Sweep AI, Linear's AI triage, and GitHub's built-in auto-triage all sell tooling in this space. Solo maintainers and small dev-tool teams routinely lose 30–60 minutes per session deciding what to look at first. A reliable resolution-probability ranker converts that time into actual work.
Target variable
A historical issue counts as a positive example (target = 1) if all four conditions hold:

state reaches "closed"
state_reason is "completed"
The closing actor is a human (not stale-bot, dependabot[bot], etc.)
All of the above occurred within 168 hours of issue.created_at

Otherwise target = 0.
Prediction
AspectValueTypeBinary classificationInputIssue payload at creation time (title, body, author features, repo features)Outputp_close_7d ∈ [0, 1], plus confidence_band, top_features, scope_caveatHorizon168 hours from issue.created_atConsumerLLM agent (Week 14+) acting on behalf of OSS maintainer
Goals

Binary classifier predicting resolution-within-168h on Python OSS repositories
API consumable by an LLM agent (typed I/O, structured confidence)
Honest out-of-domain evaluation on held-out repos
Reproducible: one command from raw scrape to deployed endpoint

Non-goals

Multi-label prediction (which specific label applies) — deferred to Week 15
RAG over similar past issues — deferred to Week 13
Per-repo personalization via fine-tuning — deferred to Week 15
Production observability and monitoring — deferred to Week 16
Code-aware features (PR diffs, codebase analysis) — out of scope (these require post-T data)
Non-Python repos — out of scope for this capstone

Success criteria
CriterionThresholdIn-domain F1≥ majority-class baseline + 0.15 absoluteIn-domain precision@5≥ 0.70Out-of-domain precision@5≥ 0.55 on held-out reposAPI latency≤ 500ms for batches of 50 issuesDocumentationModel card published with scope, training data, eval, known limits
Scope
DimensionSettingTraining repos (in-domain)huggingface/transformers, pytorch/pytorch, langchain-ai/langchainHeld-out repos (OOD)fastapi/fastapi, scikit-learn/scikit-learnLanguagesPython ecosystem onlyRepo domainsML libraries, dev tools, web frameworks, data librariesDeployment targetFastAPI service on Hugging Face SpacesConsumerLLM agent on behalf of an OSS maintainer
Feature constraints
Every candidate feature must pass the time-travel test: "Could this value be known at issue.created_at, in production, without time travel?"
Drop entirely (direct target encoding): state, state_reason, closed_at, closed_by, updated_at, linked closing PR references.
Conditional (require time-windowing):
FeatureConstraintComments count[T, T+1h] window onlyReactions countFirst-hour onlyLabelsFirst-hour only, or dropLinked PR mentionsFirst-hour only, or dropAssigneesBoolean "assigned at T" (almost always false)
Safe (use freely): title and body length and pattern features, time features (hour, day-of-week at T), author features at T (account age, follower count, prior repo contributions filtered to created_at < T), repo features snapshotted at T.
Procedural rule. Preprocessing (scalers, vocabularies, encoders) is fit on the train split only, then applied to the test split. Never fit on combined data before splitting.
Open questions

Actual class balance after Day 2 scrape (estimated ~33% positive)
Whether body-text features (TF-IDF / embeddings) add value over structural features alone
Whether the OOD precision@5 threshold (0.55) is realistic before fine-tuning
Whether to include text features in Week 11–12, or defer them to the RAG corpus in Week 13