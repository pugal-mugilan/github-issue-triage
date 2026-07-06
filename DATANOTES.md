# Data Notes

This file documents every cleaning decision in `02_clean.py` — what was
filtered, why, and what the cost was. The purpose is to ensure that
six months from now (or in an interview), every drop can be defended.

## Pipeline overview

Raw scrape (68,816 items) → cleaned, labeled dataset (≈21K rows) →
split into `cleaned_train.parquet` (3 in-domain repos) and
`cleaned_ood.parquet` (2 held-out repos).

Per-repo funnel counts are persisted in `data/processed/_clean_metadata.json`.

---

## Stage 1: Drop pull requests

**What:** Rows where the raw JSON contains a `pull_request` field are dropped.

**Why:** GitHub's `/issues` endpoint returns both issues and PRs. They
share the API shape but not the lifecycle. PRs go through review, can
be force-pushed, are merged rather than closed, and have a fundamentally
different resolution distribution. Modeling them together would inject
noise into the target signal we care about (issue resolution).

**Cost:** ≈40,000 rows dropped (most from `pytorch/pytorch` and
`huggingface/transformers`, which see heavy PR traffic).

---

## Stage 2: Drop rows outside the `since_date` window

**What:** Rows where `created_at < since_date` (540 days before the scrape)
are dropped.

**Why:** The GitHub API's `since` parameter filters by `updated_at`, not
`created_at`. An issue created 3 years ago but commented on yesterday
comes through. We want a clean *creation-date* window for the model to
learn from a current behavioral distribution. Bot behaviors, label
vocabularies, and response-time norms drift over time.

**Cost:** ≈2,000 rows dropped (long-lived issues from before the window).

---

## Stage 3: 168-hour label-observability filter

**What:** Rows where `created_at` is within 168 hours of the scrape time
are dropped, **regardless of current state**.

**Why:** Our target is "closed-as-completed within 168 hours." For an
issue opened 3 days ago, the answer is genuinely unknown — it may still
close in the next 96 hours, it may not. Keeping only the ones that
*happened* to close fast would create selection bias: training would
see all the fast-closers from the last week but none of the slow-closers
(whose outcomes aren't yet decided). The model would learn "recent
issues mostly close fast" — which is a property of the dataset, not the
world. Drop them all.

**Cost:** ≈2,000 rows dropped.

---

## Stage 4: Drop bot-authored issues

**What:** Rows where `user.type == "Bot"`, where `user.login` ends with
`[bot]`, or where `user.login` contains a known bot substring
(`dependabot`, `github-actions`, `renovate`, `pre-commit-ci`) are dropped.

**Why:** Three reasons.
  1. Bots close in minutes, not days. Including them inflates the
     positive rate and skews the model toward "looks like a bot → fast
     close," which won't generalize.
  2. Bots have no `body` or trivial bodies — they break any text feature.
  3. The Phase 2 agent will filter bots upstream. Training on them
     creates a train/production distribution mismatch.

**Cost:** ≈5,000 rows dropped (varies wildly by repo;
`huggingface/transformers` is high, `scikit-learn/scikit-learn` is low).

**Detection note:** We use heuristics (suffix + type field + known
substring list), not an allow-list. Bots come and go faster than any
maintained list, and false positives are extremely unlikely (a human
named "renovate-fan" doesn't exist).

---

## Stage 5: Drop rows with null `state_reason`

**What:** Rows where `state_reason` is `null` are dropped.

**Why:** `state_reason` distinguishes `"completed"` (resolved) from
`"not_planned"` (closed without resolution — won't fix, duplicate, etc).
This distinction is the spine of our target. A `null` `state_reason`
means we cannot responsibly label the outcome. Best to drop than to
impute.

**Cost:** ≈100-500 rows dropped. Most issues in our window were created
after GitHub introduced `state_reason` in 2022, so this is a small slice.

---

## Stage 6: Dedupe by `issue_id`

**What:** Rows with duplicate `issue_id` values are deduplicated, keeping
the first occurrence.

**Why:** Defensive. The pagination loop in `01_ingest.py` could in
theory produce a duplicate if GitHub's API returns an issue twice
across a page boundary (rare but documented). Better to dedupe
explicitly than to have a duplicate skew the loss function.

**Cost:** Usually ≈0 rows dropped, but logged regardless.

---

## Stage 7: Target label construction

**Formula:**

```
target = 1  if  state_reason == "completed"  AND  (closed_at - created_at) <= 168h
target = 0  otherwise
```

**Why this definition:** A positive label represents a successful triage
outcome — the issue got a real resolution within the actionable
window. Negative captures everything that fell short, whether that
was being too slow, getting closed as won't-fix, or being closed as
duplicate.

**Class balance:** Expected ≈30-40% positive across the dataset. Will
vary by repo. Logged in `_clean_metadata.json`.

---

## Stage 8: Type discipline

| Column | Final dtype | Reason |
|---|---|---|
| `issue_id` | `str` | IDs are labels, not numbers |
| `created_at`, `closed_at` | `datetime64[ns, UTC]` | Enables time arithmetic |
| `state_reason` | `category` | Small fixed set (~3 values) |
| `user_type`, `author_association`, `repo` | `category` | Small cardinality |
| `target` | `int` | sklearn / XGBoost convention |
| `has_milestone` | `bool` | Flag, not arithmetic |