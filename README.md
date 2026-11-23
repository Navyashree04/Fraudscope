<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=180&section=header&text=FraudScope&fontSize=50&fontColor=ffffff&fontAlignY=38&desc=ML%20fraud%20scoring%20service%20%C2%B7%20XGBoost%20%C2%B7%20FastAPI%20%C2%B7%20SHAP&descAlignY=60&descSize=16" width="100%"/>

<div align="center">

<img src="https://skillicons.dev/icons?i=py,fastapi,sklearn,docker&theme=dark" />
<br/><br/>
<img src="https://img.shields.io/badge/Model-XGBoost-00E5FF?style=flat-square" />
<img src="https://img.shields.io/badge/Explainability-SHAP%20top--3-black?style=flat-square" />
<img src="https://img.shields.io/badge/Tests-pytest%20(17)-0A9EDC?style=flat-square" />

</div>

> 📚 Short on time before an interview? Open the **Deep dive** at the bottom of this page.

# FraudScope — ML-Powered Transaction Risk Scoring Service

FraudScope is a small, production-shaped ML microservice that trains a
gradient-boosted fraud classifier offline and serves it online as a REST
API, with SHAP-based explainability on every prediction and an honest,
class-imbalance-aware evaluation report. It's the AI/ML differentiator in
a 3-project banking-domain portfolio: Projects 1 and 2 cover backend/systems
work (Java/Spring), and FraudScope demonstrates the "train offline, serve
online, explain predictions, monitor drift" pattern used in real fraud
systems — without ever claiming access to real bank data.

## ⚠️ Honesty statement (read this first)

> **This project uses the public Kaggle "Credit Card Fraud Detection"
> dataset** (anonymized, PCA-transformed features) **— or, when that
> dataset isn't available, a locally-generated synthetic stand-in with an
> identical schema, clearly labeled `synthetic-for-demo` everywhere it
> appears.** No real bank, customer, or transaction data is used anywhere
> in this project. This is a demonstration of ML engineering and serving
> practices — training pipelines, REST APIs, explainability, evaluation
> rigor, containerization — not a claim of production deployment at a
> financial institution.
>
> The build environment used to assemble this repo had no network access
> to Kaggle, so the artifacts currently in `models/` and `reports/` were
> produced from the **synthetic-for-demo fallback** (see `data/README.md`
> and `models/metadata.json` for the exact provenance flag:
> `dataset_is_synthetic`). Swapping in the real Kaggle CSV requires no
> code changes — see [Setup](#setup--run) below.

## Architecture

```
TransactionWatch (Java/Spring, Project 2)
        │  on flagged/borderline transaction
        │  POST /score  (REST, JSON)
        ▼
FraudScope (Python FastAPI service)
        │  loads trained model (joblib artifact) at startup
        │  featurizes request payload
        │  model.predict_proba() → risk score
        │  SHAP TreeExplainer → top 3 contributing features
        ▼
Response: { risk_score, top_features[], model_version, latency_ms }
        │
        ▼
TransactionWatch stores score + top features alongside the alert
```

Two clearly separated concerns:
- **Offline / training pipeline** (`src/train.py`, `src/evaluate.py`) — run
  once or on demand to produce a model artifact + metrics report.
- **Online / serving pipeline** (`app/`) — loads the artifact and serves
  predictions fast; it does **not** retrain per-request.

## Repository structure

```
fraudscope/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/            # data/README.md explains dataset provenance; CSV is gitignored
├── models/          # trained artifact + metadata.json land here (gitignored)
├── reports/         # evaluation_report.md (generated, not fabricated)
├── src/             # offline pipeline: config, train, evaluate, explain, features, synthetic_data, logging_utils
├── app/             # online service: FastAPI app, schemas, routes (score/health/retrain)
├── tests/           # pytest suite
└── scripts/         # example_client.py — simulated TransactionWatch → FraudScope call
```

## Setup & run

### 1. Install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) get the real dataset

Download `creditcard.csv` from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at
`data/creditcard.csv`. If you skip this step, `src/train.py` will
auto-generate a schema-matching **synthetic-for-demo** CSV in its place —
see `data/README.md` for details.

### 3. Train

```bash
python -m src.train
```

Trains an `XGBClassifier` with `scale_pos_weight` class-imbalance handling,
sanity-checks a SHAP `TreeExplainer` against the trained model, and saves
`models/fraud_model_v1.joblib` + `models/metadata.json`.

### 4. Evaluate

```bash
python -m src.evaluate
```

Computes ROC-AUC, PR-AUC, precision/recall/F1 at multiple thresholds on the
held-out test split, and writes `reports/evaluation_report.md`.

### 5. Test

```bash
pytest -v
```

Note: several tests (the model-metrics regression floor, the explainer
tests, the score-endpoint tests) run against the actual trained artifact,
so `python -m src.train` must be run first — CI should train the model as
a prerequisite step before running the test suite.

### 6. Serve

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Docker

```bash
docker build -t fraudscope .
docker run -p 8000:8000 -v $(pwd)/models:/srv/fraudscope/models fraudscope
```

or, with compose:

```bash
docker compose up --build
```

## API contract

### `POST /score`

Request:

```json
{
  "transaction_id": "txn_watch_00042",
  "Time": 39016.0,
  "Amount": 8.02,
  "V1": -1.165, "V2": -0.103, "V3": -1.520, "V4": -5.234,
  "V5": 1.910,  "V6": 0.697,  "V7": 2.323,  "V8": 3.508,
  "V9": -1.946, "V10": -0.668, "V11": -5.256, "V12": 0.602,
  "V13": -0.683, "V14": 2.582, "V15": -0.315, "V16": -0.043,
  "V17": -5.355, "V18": -3.293, "V19": 0.944, "V20": 1.161,
  "V21": 0.185, "V22": 2.122, "V23": -0.996, "V24": -1.470,
  "V25": 0.704, "V26": 0.602, "V27": -2.241, "V28": 0.110
}
```

Response:

```json
{
  "transaction_id": "txn_watch_00042",
  "risk_score": 0.999988,
  "risk_band": "high",
  "top_features": [
    { "feature": "V17", "contribution": 7.141281, "direction": "increases risk" },
    { "feature": "V11", "contribution": 1.230214, "direction": "increases risk" },
    { "feature": "V4",  "contribution": 1.066011, "direction": "increases risk" }
  ],
  "model_version": "v1",
  "scored_at": "2026-08-21T18:23:38.407663Z",
  "latency_ms": 2.132
}
```

(This example was captured live against a fraud-labeled row from the
synthetic-for-demo dataset — the SHAP explainer correctly surfaces the
features that were deliberately shifted to create the fraud signal in
that generator.)

### `GET /health`

```json
{ "status": "ok", "model_version": "v1", "uptime_s": 1.153 }
```

### `POST /retrain` (demo stub only)

Returns `202 Accepted` immediately and runs `src.train.train()` as a
FastAPI background task, then hot-reloads the in-process model singleton.
**This is a demo convenience, not a production retraining system** — see
Limitations below.

## Evaluation results

Full report: [`reports/evaluation_report.md`](reports/evaluation_report.md).
Headline numbers from the current artifact (synthetic-for-demo dataset,
10,000-row stratified held-out test split, 17 fraud rows):

| Metric | Value |
|---|---|
| ROC-AUC | 1.0000 |
| PR-AUC (average precision) | 0.9866 |
| Precision @ 0.5 | 1.0000 |
| Recall @ 0.5 | 0.9412 |

**Why PR-AUC matters more than ROC-AUC here:** at ~0.17% fraud prevalence,
a model that predicts "legitimate" for nearly everything can still post a
high ROC-AUC (and >99.8% raw accuracy), because ROC-AUC is dominated by the
abundant negative class. PR-AUC isolates precision/recall trade-offs on the
fraud class specifically — the number a fraud-ops team actually cares
about. Note that these particular numbers are inflated by the synthetic
generator's clean, strongly-separated signal; on the real Kaggle dataset,
expect PR-AUC in the 0.80–0.87 range, which is still strong but more
representative of real-world separability.

## Explainability

Every `/score` response includes the top-3 SHAP-attributed features driving
that specific prediction, with a direction (`increases risk` /
`decreases risk`) derived from the sign of the SHAP value — not just a
global feature-importance ranking. This matters both for analyst trust
(a fraud analyst can see *why* a transaction was flagged, not just a bare
number) and for the kind of explainability regulators increasingly expect
around automated lending/fraud decisions.

## Design decisions & trade-offs

- **XGBoost over a neural net.** Tabular data with strong, sometimes
  nonlinear-but-low-dimensional interactions (already PCA-compressed here)
  is the classic case where gradient-boosted trees match or beat deep
  learning, train in seconds instead of hours, and are directly explainable
  with SHAP's `TreeExplainer` — a much cheaper explainability story than
  gradient-based attribution on a neural net.
- **`scale_pos_weight` over resampling (SMOTE, undersampling).** Weighting
  the loss avoids duplicating or discarding real rows, which is simpler to
  reason about and doesn't risk synthesizing implausible minority-class
  samples via SMOTE on already-PCA'd features.
- **Threshold tuning is a business decision, not a fixed constant.** The
  evaluation report shows precision/recall at three thresholds precisely
  because the "right" cutoff depends on the relative cost of a missed
  fraud versus an analyst wasting time on a false alarm — that's a
  fraud-ops policy call, not something to hardcode.
- **What I'd change for real production use:**
  - A real feature store, so online (serving-time) and offline
    (training-time) feature computation can't silently drift apart.
  - Online/offline skew monitoring — the `logging_utils.py` structured
    JSON log is a stub for this, not a working monitoring pipeline.
  - Model versioning + rollback (the `model_version` field is a start, but
    there's no registry, no canary rollout, no automatic rollback here).
  - A/B testing of decision thresholds against real business outcomes
    rather than a single fixed default.
  - `/retrain` behind an actual async job queue with approval gates,
    instead of a synchronous background task that mutates the serving
    model in place.

## Integration with Project 2 (TransactionWatch)

FraudScope is deliberately polyglot: a Java/Spring service calls this
Python service over a plain REST/JSON boundary rather than forcing
everything into one stack — a common real-world pattern for isolating ML
workloads in Python behind a language-agnostic API.

**Python client** (see [`scripts/example_client.py`](scripts/example_client.py)
for the full runnable version):

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_1", "Time": 40000, "Amount": 149.62, "V1": 0, ... "V28": 0}'
```

**Java (`java.net.http.HttpClient`) snippet**, illustrating how
TransactionWatch would call FraudScope:

```java
HttpClient client = HttpClient.newHttpClient();
String body = """
    {"transaction_id":"txn_1","Time":40000,"Amount":149.62,"V1":0,"V2":0, ... "V28":0}
    """;
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://fraudscope:8000/score"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(body))
    .build();
HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
// response.body() -> { risk_score, risk_band, top_features[], model_version, ... }
```

## Limitations — what's NOT real here

- **Dataset:** public or synthetic-for-demo only — no real bank, card
  network, or customer transaction data, ever.
- **No live transaction stream:** there is no real-time ingestion pipeline;
  `/score` scores whatever payload it's given, on demand.
- **No real-time feature engineering:** the served features are exactly
  the dataset's columns (`Time`, `V1`–`V28`, `Amount`); there's no online
  feature computation, joins, or aggregation pipeline.
- **`/retrain` is a demo stub**, not a production job queue — no approval
  gates, no atomic artifact swap, no versioned rollback.
- **Drift monitoring is a logging stub** (`src/logging_utils.py`) — the
  structured JSON schema is designed to *feed* a real pipeline like
  Evidently or a feature-store drift dashboard, but no such pipeline is
  wired up here.
- **TransactionWatch (Project 2) is not built in this repo** — only a
  clean REST contract and example clients demonstrating how it would call
  FraudScope.

## Interview talking points

- *"I trained and served a real gradient-boosted classifier on a public
  fraud dataset — this demonstrates the same engineering pattern (train
  offline, serve online, explain predictions, monitor drift) used in
  production fraud systems, without claiming access to real bank data."*
- *"Class imbalance is the central technical challenge in fraud detection
  — I handled it via `scale_pos_weight` and evaluated with PR-AUC rather
  than accuracy, since accuracy is misleading at <1% positive rate."*
- *"Explainability isn't an afterthought — SHAP values on every scored
  transaction let a fraud analyst see why a transaction was flagged,
  which matters both for analyst trust and regulatory explainability
  requirements in lending/fraud decisions."*
- *"The service is deliberately polyglot: Project 2's Java service calls
  this Python service over REST, mirroring how real fintechs often
  isolate ML workloads in Python microservices behind a REST boundary
  rather than forcing everything into one stack."*
- **"Is this real bank data?"** No — and that's fine. This project isn't
  a claim of data access; it's a demonstration of engineering practice:
  a properly separated train/serve pipeline, honest evaluation under
  severe class imbalance, per-prediction explainability, and a clean
  polyglot service boundary. Anyone can download the same public Kaggle
  dataset and reproduce these results end-to-end.


---

<details>
<summary><b>📚 Deep dive: interview notes (click to expand)</b></summary>

### 1. Elevator pitch (30 seconds)
FraudScope is a fraud-scoring ML microservice. An XGBoost model is trained **offline**, saved as an artifact, and served **online** through a FastAPI `POST /score` endpoint. Every prediction comes back with its **top 3 SHAP reasons**, so an analyst sees *why* a transaction was flagged, not just a number. It's designed to sit behind TransactionWatch (Java) over a plain REST/JSON boundary.

### 2. Architecture and flow
```text
OFFLINE (src/)                                  ONLINE (app/)
data → features.py → train.py (XGBoost)         FastAPI starts → loads model + SHAP explainer once
     → evaluate.py → reports/evaluation_report  POST /score
     → models/fraud_model_v1.joblib + metadata     → validate (pydantic schemas)
                                                   → single_transaction_to_vector()
                                                   → predict_proba → risk_score + risk_band
                                                   → SHAP TreeExplainer → top 3 features + direction
                                                   → {risk_score, top_features, model_version, latency_ms}
                                                GET /health · POST /retrain (background task, demo only)
```

**Data:** the Kaggle "Credit Card Fraud Detection" schema (`Time`, `Amount`, `V1`–`V28` PCA features), with fraud at about **0.17%**. The committed artifact was trained on a **synthetic stand-in with the identical schema** (50,000 rows, 85 frauds), clearly flagged `dataset_is_synthetic: true` in `metadata.json`. Swapping in the real CSV needs no code changes.

### 3. Results (synthetic data, stratified 80/20 split, 10,000 test rows, 17 frauds)
| ROC-AUC | PR-AUC | Precision @0.5 | Recall @0.5 | F1 @0.5 |
|:--:|:--:|:--:|:--:|:--:|
| 1.0000 | 0.9866 | 1.0000 | 0.9412 | 0.9697 |

These are inflated by the synthetic data's clean signal. On the real Kaggle data, expect PR-AUC of roughly 0.80–0.87.

### 4. Key design decisions and trade-offs
| Decision | Why |
|:--|:--|
| **XGBoost**, not a neural net | Tabular, low-dimensional data is where boosted trees win; trains in seconds; exact, cheap explanations with `TreeExplainer` |
| **`scale_pos_weight` = negatives / positives**, not SMOTE | Reweights the loss instead of inventing synthetic minority rows on already-PCA'd features |
| **PR-AUC** as the headline metric | At 0.17% fraud, "always legit" gets 99.8% accuracy and a decent ROC-AUC; PR-AUC measures performance on the fraud class itself |
| **Threshold is a business choice** | The report shows precision/recall at 0.3 / 0.5 / 0.7, because the right cutoff depends on the cost of missed fraud vs analyst time |
| **Per-prediction SHAP**, not global importance | Analysts and regulators need the reason for *this* decision |
| **Load the model once at startup** | Scoring takes about 2 ms; no per-request disk reads |
| **Python behind REST** | Keeps the ML stack separate from the Java services; a common real-world polyglot pattern |

### 5. Where the important code lives
- `src/train.py`: stratified split, `scale_pos_weight`, training, saving the artifact
- `src/evaluate.py`: metrics and the threshold table → `reports/evaluation_report.md`
- `src/explain.py`: SHAP top-k with direction
- `src/features.py`: column validation and payload → feature vector
- `app/routes/score.py`, `health.py`, `retrain.py` · `app/schemas.py`
- `tests/`: 17 pytest tests (score endpoint, health, features, explain, model metrics)

### 6. Limitations and what I'd do next
- Train and report on the **real** Kaggle dataset.
- A **feature store** so training and serving features can't drift apart; **online/offline skew monitoring**.
- A **model registry** with versioning, canary rollout and rollback.
- `/retrain` behind a real job queue with approval, not an in-place background task.
- Actually wiring TransactionWatch → FraudScope (the contract and a Java client snippet are in this README).

### 7. Likely interview questions
- **Why is accuracy useless here?** With 0.17% fraud, predicting "legit" for everything scores 99.83%.
- **ROC-AUC vs PR-AUC?** ROC-AUC is dominated by the huge negative class; PR-AUC focuses on how well you find the rare positives.
- **How did you handle class imbalance?** `scale_pos_weight` (negatives ÷ positives) in XGBoost, plus a stratified split so the test set keeps the fraud ratio.
- **How does SHAP work in one sentence?** It splits a single prediction into additive contributions from each feature (Shapley values from game theory); `TreeExplainer` computes them exactly and quickly for tree models.
- **How would you pick the threshold?** From the cost of a false negative (fraud loss) vs a false positive (review time and customer friction); the report gives the trade-off table.
- **What's training/serving skew?** Features computed differently at training time and serving time; a feature store prevents it.

</details>
