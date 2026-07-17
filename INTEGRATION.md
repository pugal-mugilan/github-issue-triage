# INTEGRATION.md — Agent Tool Contract

**Service:** GitHub Issue Triage Classifier
**Version:** 0.1.0
**Last updated:** 2026-07-18

---

## What this tool does

Predicts whether a GitHub issue will be resolved within 7 days, based on 18 structural features (title/body text signals, timing, repo, author association). Returns a probability, confidence band, and scope caveat so the calling agent can decide whether to trust the prediction.

---

## How to start the service

```bash
# 1. Clone the repo
git clone https://github.com/pugal-mugilan/github-issue-triage.git
cd github-issue-triage

# 2. Build and run via Docker
docker compose up --build

# Service is now available at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

**Cold start:** First request after container start takes ~2–3 seconds (model loading). Subsequent requests are fast (p50 = 6ms individual, p50 = 123ms for 50-issue batch).

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check — returns 200 if model is loaded, 503 if not |
| GET | `/model/info` | Model metadata — version, features, train metrics, known limitations |
| POST | `/predict` | Predict for a single issue |
| POST | `/predict/batch` | Predict for multiple issues at once |

---

## `/predict` — Single issue prediction

### Request

**Method:** POST
**Content-Type:** application/json

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | yes | Issue title text |
| `body` | string | no (default: `""`) | Issue body/description |
| `repo` | string | yes | Full repo name, e.g. `"pytorch/pytorch"` |
| `author_association` | string | yes | One of: `COLLABORATOR`, `CONTRIBUTOR`, `MEMBER`, `NONE` |
| `user_login` | string | yes | GitHub username of the issue author |
| `created_at` | string (ISO 8601) | yes | Issue creation timestamp in UTC |

### Example request (curl)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Bug: tokenizer crashes on empty string",
    "body": "When passing an empty string to AutoTokenizer, it throws IndexError.\n\n```python\ntokenizer = AutoTokenizer.from_pretrained(\"bert-base-uncased\")\ntokenizer(\"\")\n```\n\nTraceback:\n```\nIndexError: list index out of range\n```",
    "repo": "huggingface/transformers",
    "author_association": "NONE",
    "user_login": "some-user",
    "created_at": "2026-07-18T10:30:00Z"
  }'
```

### Example request (Python)

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "title": "Bug: tokenizer crashes on empty string",
    "body": "When passing an empty string to AutoTokenizer, it throws IndexError.",
    "repo": "huggingface/transformers",
    "author_association": "NONE",
    "user_login": "some-user",
    "created_at": "2026-07-18T10:30:00Z",
})

prediction = response.json()
```

### Response

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | boolean | `true` = likely resolved within 7 days, `false` = unlikely |
| `p_close_7d` | float | Probability of resolution within 7 days, range [0, 1] |
| `confidence_band` | string | `"low"` / `"medium"` / `"high"` — discrete signal for decision-making |
| `top_features` | list[string] | Top 3 features that drove this prediction |
| `scope_caveat` | string or null | Warning when prediction reliability is reduced |
| `model_version` | string | Model version string |

### Example response

```json
{
  "predicted_class": true,
  "p_close_7d": 0.6823,
  "confidence_band": "high",
  "top_features": ["title_length", "body_length", "body_word_count"],
  "scope_caveat": null,
  "model_version": "0.1.0"
}
```

### Confidence band thresholds

| Band | Condition | Agent action |
|------|-----------|-------------|
| `high` | probability ≥ 0.65 or ≤ 0.35 | Trust the prediction |
| `medium` | 0.35 < probability < 0.45 or 0.55 < probability < 0.65 | Use with caution, consider additional signals |
| `low` | 0.45 ≤ probability ≤ 0.55 | Near decision boundary — do not auto-act, escalate or gather more context |

---

## `/predict/batch` — Batch prediction

### Request

**Method:** POST
**Content-Type:** application/json

```json
{
  "issues": [
    { "title": "...", "body": "...", "repo": "...", "author_association": "...", "user_login": "...", "created_at": "..." },
    { "title": "...", "body": "...", "repo": "...", "author_association": "...", "user_login": "...", "created_at": "..." }
  ]
}
```

Each item in `issues` has the same schema as the single `/predict` request.

### Response

```json
{
  "predictions": [
    { "predicted_class": true, "p_close_7d": 0.6823, "confidence_band": "high", "top_features": [...], "scope_caveat": null, "model_version": "0.1.0" },
    { "predicted_class": false, "p_close_7d": 0.3102, "confidence_band": "high", "top_features": [...], "scope_caveat": "Repo not in training data — prediction reliability is reduced.", "model_version": "0.1.0" }
  ],
  "total": 2
}
```

### Latency

| Metric | Value |
|--------|-------|
| Individual p50 | 6 ms |
| Individual p95 | 7 ms |
| Individual p99 | 7.7 ms |
| Batch (50 issues) p50 | 123 ms |

All measured inside Docker container on CPU.

---

## `/health` — Liveness check

```bash
curl http://localhost:8000/health
```

**200 response:** `{"status": "ok"}` — model loaded, ready for predictions.
**503 response:** `{"status": "error", "detail": "model not loaded"}` — container is up but model failed to load.

---

## `/model/info` — Model metadata

```bash
curl http://localhost:8000/model/info
```

Returns model version, architecture description, feature list, training repos, train metrics, and known limitations. Useful for the agent to introspect what it's calling before making decisions based on predictions.

---

## Error handling

| Status code | Meaning | Agent action |
|-------------|---------|-------------|
| 200 | Success | Use the prediction |
| 400 | Bad request (business logic error) | Log and skip this issue |
| 422 | Validation error (missing/wrong-type field) | Fix the request payload |
| 405 | Wrong HTTP method (e.g. GET on /predict) | Use POST |
| 500 | Internal server error | Retry once, then skip |
| 503 | Model not loaded | Wait and retry `/health` before sending predictions |

### Example 422 response (missing required field)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "repo"],
      "msg": "Field required",
      "input": { "title": "some title" }
    }
  ]
}
```

---

## Function-calling spec (for LLM tool use)

The following JSON schema can be used directly in an LLM function-calling definition (e.g. OpenAI `tools`, Anthropic `tool_use`, LangChain `StructuredTool`):

```json
{
  "name": "predict_issue_resolution",
  "description": "Predicts whether a GitHub issue will be resolved within 7 days based on structural features. Returns probability, confidence band, and scope warnings. Do NOT auto-act on 'low' confidence predictions.",
  "parameters": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "Issue title text"
      },
      "body": {
        "type": "string",
        "description": "Issue body/description text (may be empty)"
      },
      "repo": {
        "type": "string",
        "description": "Full repo name, e.g. 'pytorch/pytorch'"
      },
      "author_association": {
        "type": "string",
        "enum": ["COLLABORATOR", "CONTRIBUTOR", "MEMBER", "NONE"],
        "description": "Author's relationship to the repo"
      },
      "user_login": {
        "type": "string",
        "description": "GitHub username of the issue author"
      },
      "created_at": {
        "type": "string",
        "format": "date-time",
        "description": "Issue creation timestamp, ISO 8601 UTC"
      }
    },
    "required": ["title", "repo", "author_association", "user_login", "created_at"]
  }
}
```

---

## Known limitations (from MODEL_CARD.md)

These should be treated as **agent-facing warnings** — conditions where the prediction should not be auto-acted upon:

1. **In-domain P@5 = 0.60 (below 0.70 target)** — When ranking the top 5 "most likely to close" issues, 2 out of 5 are wrong. The model is better at binary classification than ranking.

2. **Langchain slice has lower precision** — If `repo` is `"langchain-ai/langchain"`, expect more false positives. The `scope_caveat` field flags this automatically.

3. **Out-of-distribution repos** — If `repo` is not one of `huggingface/transformers`, `langchain-ai/langchain`, or `pytorch/pytorch`, the model has never seen that repo's patterns. The `scope_caveat` field flags this automatically.

4. **`author_prior_count` defaults to 0 at inference** — The model was trained with author history features, but the serving path defaults all authors to 0 (first-time). This slightly reduces prediction accuracy for repeat authors.

5. **Probability calibration not formally tested** — Use `p_close_7d` as a relative ranking signal, not an absolute likelihood.

---

## Trained on

| Repo | Issues |
|------|--------|
| `huggingface/transformers` | majority of training data |
| `langchain-ai/langchain` | included (weaker slice) |
| `pytorch/pytorch` | included |

---

## Version history

| Version | Date | Notes |
|---------|------|-------|
| 0.1.0 | 2026-07-18 | Initial release — 18 structural features, NN classifier, Docker deployment |