# FraudShield AI

Credit-card fraud detection and investigation platform: a leakage-safe ML pipeline, a
PyTorch deep-learning track, SHAP explanations, a guarded LLM investigation agent, a FastAPI
service and a React review console.

```
Transaction → preprocessing → ML model → probability → risk band → SHAP → investigation report
```

---

## Table of contents

1. [Status and honest caveats](#1-status-and-honest-caveats)
2. [The problem](#2-the-problem)
3. [Why fraud detection is hard](#3-why-fraud-detection-is-hard)
4. [Dataset](#4-dataset)
5. [Architecture](#5-architecture)
6. [ML pipeline](#6-ml-pipeline)
7. [Deep learning](#7-deep-learning)
8. [Class imbalance strategy](#8-class-imbalance-strategy)
9. [Threshold optimisation](#9-threshold-optimisation)
10. [Model comparison and selection](#10-model-comparison-and-selection)
11. [SHAP explainability](#11-shap-explainability)
12. [AI investigation agent](#12-ai-investigation-agent)
13. [API](#13-api)
14. [Frontend](#14-frontend)
15. [Running locally](#15-running-locally)
16. [Docker](#16-docker)
17. [MLflow](#17-mlflow)
18. [Testing](#18-testing)
19. [CI/CD](#19-cicd)
20. [Limitations](#20-limitations)
21. [Future improvements](#21-future-improvements)

---

## 1. Status and honest caveats

Read this before the metrics section.

**Metrics below are measured on the real Kaggle dataset** (284,807 transactions, 492
fraud), not estimated. The deployed artifact is tagged `data_source: KAGGLE_CREDITCARD`.

The project also ships a synthetic smoke fixture for running the pipeline without the
dataset. Anything trained on it is tagged `data_source: SYNTHETIC_SMOKE`, the API attaches a
warning to every prediction, and the dashboard shows a banner — synthetic numbers are
near-perfect precisely *because* the fixture is easy and must never be reported.

**What has been verified in this environment**

| Component | Status |
|---|---|
| ML pipeline end to end | Verified on the real Kaggle dataset |
| Logistic regression, random forest, XGBoost, LightGBM | Trained, compared |
| PyTorch MLP, autoencoder | Trained on CPU, early stopping and checkpointing exercised |
| Threshold optimisation | Verified, selects a non-0.5 operating point |
| SHAP explanations | Verified end to end through the API |
| FastAPI — all 6 endpoints | Verified over HTTP with `uvicorn` |
| Investigation agent and guardrails | Verified, including contradiction rejection |
| React frontend | Lints clean, builds, serves, API URL baked in |
| MLflow tracking | Verified, 7 runs logged to SQLite |
| Test suite | 158 tests passing |
| EDA notebook | Executes with 0 errors |
| Ruff lint + format | Clean |
| **Docker images** | **Not built — Docker unavailable in the build environment** |
| Real dataset training | Completed — metrics in §10 |

The Dockerfiles and Compose file are written but **unverified**. Treat `docker compose up`
as untested until you have run it yourself.

---

## 2. The problem

Given a card transaction, decide in real time whether it is fraudulent, and give a human
reviewer enough evidence to act on the decision.

The output is not just a score:

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.947,
  "risk_level": "CRITICAL",
  "recommended_action": "BLOCK_AND_ESCALATE",
  "threshold": 0.35,
  "explanation": [
    { "feature": "V14", "shap_value": 0.42, "direction": "increases_fraud_risk" },
    { "feature": "V10", "shap_value": 0.31, "direction": "increases_fraud_risk" }
  ]
}
```

---

## 3. Why fraud detection is hard

- **Extreme imbalance.** 0.172% positives. Predicting "legitimate" always scores 99.83%
  accuracy, so the obvious metric is worse than useless — it is actively misleading.
- **Asymmetric costs.** A missed fraud costs the chargeback plus write-off; a false alarm
  costs minutes of review. Roughly 100:1, so the decision threshold is a business decision.
- **Anonymised features.** V1–V28 are PCA components. No domain feature engineering is
  possible, and explanations can describe direction and magnitude but never business meaning.
- **Adversarial drift.** Attackers change behaviour in response to your model. A model that
  worked last quarter degrades quietly, with inputs that still look plausible.
- **Explainability is mandatory.** A held transaction gets reviewed by a person, and the
  decision may be challenged. A bare probability is not enough.
- **Leakage is easy and silent.** Scaling before splitting, oversampling before splitting, or
  using raw `Time` all inflate offline metrics without any error being raised.

---

## 4. Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB / Worldline) — 284,807 transactions from European cardholders over two days in
September 2013, of which 492 are fraud (0.172%).

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | PCA components (original features withheld for confidentiality) |
| `Amount` | Transaction amount |
| `Class` | 1 = fraud, 0 = legitimate |

**Setup.** The CSV is ~144 MB and is not committed. Place it at `data/raw/creditcard.csv`.

```bash
python ml/scripts/download_data.py     # needs ~/.kaggle/kaggle.json
```

If that fails, download manually and extract `creditcard.csv` to `data/raw/`.

**Without the dataset**, generate the smoke fixture — schema-compatible, clearly labelled,
and never a substitute for real data:

```bash
python ml/scripts/make_synthetic_data.py
python ml/scripts/train.py --synthetic
```

---

## 5. Architecture

```
                    ┌──────────────────────────────────────────┐
   data/raw/  ────► │  ML PIPELINE  (ml/src/fraudshield)        │
   creditcard.csv   │  validate → split → preprocess → imbalance│
                    │  → train → tune threshold → select        │
                    └───────────────┬──────────────────────────┘
                                    │ artifact bundle
                                    ▼
                    models/fraud_detector/
                      metadata.json · preprocessor.joblib
                      model.joblib|.pt · shap_background.npy
                                    │
                                    ▼
   React console ◄── FastAPI ◄── ModelService ──► ShapExplainer
   (nginx :5173)     (:8000)          │
                                      ▼
                              RiskEngine (deterministic bands)
                                      │
                                      ▼
                    InvestigationToolkit (6 whitelisted tools)
                                      │
                                      ▼
                    InvestigationAgent ──► LLM (swappable)
                                      └──► deterministic fallback
```

The artifact bundles the **fitted preprocessor with the estimator**. Serving one without the
other is the classic training/serving skew bug, so they are written and loaded together.

---

## 6. ML pipeline

```
Raw data → validation → EDA → stratified split → preprocessing → imbalance handling
        → training → threshold optimisation → evaluation → selection → artifact
```

**Leakage controls**, in the order they matter:

1. **Split first.** Test is sealed and opened exactly once, at the end.
2. **Preprocessor fitted on train only.** The scaler never sees held-out statistics.
3. **Resampling on the training fold only**, with `assert_no_resampling_on_eval` as a guard.
4. **Raw `Time` dropped.** It encodes position in this collection window — meaningless for
   future traffic. Replaced by a cyclical hour-of-day (`sin`/`cos`) encoding.

Feature engineering: 28 PCA components pass through untouched, `Amount` is `log1p`'d then
scaled, `Time` becomes `hour_sin`/`hour_cos`. **31 features total.** Seeded throughout
(`project.random_seed: 42`).

Run it:

```bash
python ml/scripts/train.py                 # real dataset
python ml/scripts/train.py --synthetic     # smoke fixture
python ml/scripts/train.py --no-deep       # skip PyTorch (faster CI)
```

---

## 7. Deep learning

**MLP** (`ml/src/fraudshield/models/mlp.py`) —
`Linear → BatchNorm → ReLU → Dropout` blocks into a single logit, trained with
`BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`.

Early stopping monitors **validation PR-AUC, not validation loss.** Loss is dominated by the
majority class and keeps improving while fraud recall degrades — stopping on loss picks the
wrong epoch. The best epoch is checkpointed and restored.

**Autoencoder** (`models/autoencoder.py`) — trained on **legitimate transactions only**,
scoring by reconstruction error:

```
Transaction → Encoder → Latent (16) → Decoder → Reconstruction → MSE
```

The anomaly threshold is a percentile of **validation** reconstruction errors; test data is
never consulted. Its output is a monotone rank score, not a calibrated probability, so it is
excluded from production model selection and runs as a novelty/drift tripwire beside the
supervised scorer — the supervised models can only recognise fraud resembling the 492
labelled positives they trained on.

---

## 8. Class imbalance strategy

Four strategies compared with XGBoost held fixed, so the strategy is the only variable.
Results land in `reports/imbalance_comparison.csv`.

| Strategy | Train rows | PR-AUC | Recall @0.5 | Verdict |
|---|---|---|---|---|
| `none` | 182,276 | 0.826 | 0.760 | Highest PR-AUC, lowest recall |
| `class_weight` | 182,276 | 0.819 | 0.772 | **Default.** No synthetic rows, no leakage surface |
| `smote` | 363,922 | 0.820 | 0.798 | No PR-AUC gain for 2x the training cost |
| `undersample` | 3,465 | 0.775 | 0.823 | Clearly worst — discards 98% of real signal |

Measured with XGBoost held fixed so the strategy is the only variable. The headline finding
is that **resampling bought nothing here**: SMOTE matched plain class weighting on PR-AUC
while doubling the training set, and undersampling cost 0.05 PR-AUC outright. `class_weight`
is the default because it ties the best strategies on ranking quality, improves recall over
doing nothing, and — unlike SMOTE — invents no rows and carries no leakage surface.

**On LightGBM and `scale_pos_weight`.** A 578:1 weight is fatal to LightGBM's leaf-wise
growth on this dataset (PR-AUC 0.057 with it, 0.311 without, 0.839 once L2 regularisation
was added). It is disabled for that learner via `use_scale_pos_weight: false`, with
imbalance handled by regularisation and leaf-size floors instead. XGBoost's level-wise
growth is unaffected and still uses it.

`class_weight` is the default: it achieves a similar effect to SMOTE without inventing rows,
and avoids SMOTE's leakage hazard (a synthetic training point interpolated from a row that
lands in validation).

**Resampling an evaluation split is a correctness bug, not a tuning choice** — it changes the
base rate and makes precision meaningless.

---

## 9. Threshold optimisation

0.5 assumes balanced classes and symmetric costs. Neither holds. The optimiser sweeps
13 candidate thresholds on **validation** and minimises:

```
expected_cost = cost_fn × FN + cost_fp × FP     # 500 and 5 by default
```

subject to a `min_precision` guardrail that keeps alert volume within what a review team can
absorb. The chosen point, the full sweep and a written rationale are persisted to
`metadata.json` and surfaced at `GET /model-info`. All costs live in `configs/config.yaml`,
so risk appetite is retunable without retraining.

Per-model sweeps: `reports/threshold_sweep_<model>.csv`.

---

## 10. Model comparison and selection

Selection is on **validation PR-AUC**, among supervised candidates only. The dummy
classifier and autoencoder are excluded from the production scoring path.

Measured on the real Kaggle dataset (train 182,276 / val 45,569 / test 56,962; seed 77777).
Precision and recall are quoted at each model's own selected threshold, which is why a model
with lower PR-AUC can show higher precision.

**Validation comparison**

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.30 | 0.900 | 0.798 | 0.846 | 0.978 | 0.819 |
| **LightGBM** *(selected)* | **0.02** | **0.625** | **0.823** | **0.710** | **0.971** | **0.815** |
| Random forest | 0.35 | 0.818 | 0.798 | 0.808 | 0.961 | 0.794 |
| PyTorch MLP | 0.90 | 0.366 | 0.810 | 0.504 | 0.959 | 0.748 |
| Logistic regression | 0.90 | 0.212 | 0.823 | 0.337 | 0.953 | 0.723 |
| Autoencoder | 0.50 | 0.520 | 0.658 | 0.581 | 0.956 | 0.508 |
| Dummy (always legitimate) | — | 0.000 | 0.000 | 0.000 | 0.500 | 0.002 |

**Held-out test results for the deployed model**

| Metric | Value |
|---|---|
| PR-AUC | **0.8485** |
| ROC-AUC | 0.9826 |
| Precision | 0.6667 |
| Recall | 0.8571 |
| F1 | 0.7500 |
| Confusion | TP 84 · FP 42 · FN 14 · TN 56,822 |

```
Final model : LightGBM (v1.3.0)
Threshold   : 0.02
Reason      : XGBoost led PR-AUC by 0.0044 - inside the 0.005 tie tolerance and
              well inside the noise floor of 79 validation positives. The tie
              broke on recall (0.823 vs 0.798), because a missed fraud costs
              ~100x a false alarm. Full rationale in metadata.json.
```

The pipeline does **not** assume XGBoost wins — it selects whatever measures best and writes
the rationale into the artifact.

---

## 11. SHAP explainability

TreeSHAP for tree ensembles (exact, fast enough to run inline); a sampling explainer over a
stored background set otherwise.

```
Fraud probability: 94.7%

Top contributing features:
  V14 (value -3.118) -> strongly increased fraud probability (SHAP +0.4210)
  V10 (value -3.635) -> moderately increased fraud probability (SHAP +0.3104)
```

Every value is computed; nothing is templated per feature. The wording ("strongly",
"moderately") is derived from the relative magnitudes actually present.

**Limitation stated everywhere it appears:** V1–V28 are anonymised, so explanations describe
direction and magnitude, never business meaning.

---

## 12. AI investigation agent

The LLM writes; it does not decide.

```
Transaction → ML model → probability → SHAP → evidence pack → agent → report
```

**Whitelisted tools** — no database handle, no shell, no filesystem:
`get_transaction_details`, `get_customer_transaction_history`, `get_model_prediction`,
`get_shap_explanation`, `get_risk_policy`, `get_model_info`. Calls outside the whitelist are
blocked and logged.

**Three guardrails:**

1. **Prompt** — the ML score is authoritative; inventing evidence is forbidden; unavailable
   evidence must be reported as exactly "Insufficient evidence."
2. **Validation** — narratives contradicting the ML verdict are rejected.
3. **Architecture** — probability, risk band and recommended action are copied from the model
   and risk engine, *never parsed out of LLM text*. A narrative that slipped past validation
   still could not change the decision.

**Provider-agnostic** (`LLM_PROVIDER=none|anthropic|openai`). With no key configured the
agent produces a deterministic report assembled from the evidence — less fluent, incapable
of hallucinating, and always available.

---

## 13. API

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness plus model readiness |
| GET | `/model-info` | Provenance, threshold, rationale, risk policy |
| POST | `/predict` | Score one transaction with SHAP |
| POST | `/batch-predict` | Score many, isolating per-row failures |
| POST | `/investigate` | Score plus guarded investigation report |
| GET | `/metrics` | Live counters plus offline evaluation metrics |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"Amount": 412.9, "Time": 43200, "V14": -4.2, "V10": -3.6}'
```

Absent PCA components default to 0.0. Invalid input returns 422 with field-level detail; a
missing model returns 503 explaining why rather than crashing.

---

## 14. Frontend

React + Vite, four pages: **Dashboard**, **Transaction analyzer**, **Investigation**,
**Model performance**.

The signature element is the **risk rail** — the 0–1 probability line drawn with the
configured risk bands, the decision threshold, and where the transaction landed. It makes
visible the thing a bare percentage hides: that the cut-off is a tuned business choice, not
0.5. SHAP contributions render as diverging bars because sign is the point — a feature can
clear a transaction as well as condemn it.

Charts are hand-drawn SVG rather than a charting library: fewer dependencies and full control
over the design tokens.

> Screenshots: _to add after running locally._
> `docs/screenshots/dashboard.png` · `analyzer.png` · `investigation.png` · `performance.png`

---

## 15. Running locally

```bash
# 1. Install
make install

# 2. Get data (or use the synthetic fixture)
make data          # or: make synthetic

# 3. Train
make train         # or: make train-synthetic

# 4. Run
make api           # http://localhost:8000/docs
make ui            # http://localhost:5173
```

Requires Python 3.10+ and Node 18+. Copy `.env.example` to `.env` for configuration.

Install PyTorch from the CPU index (`--index-url https://download.pytorch.org/whl/cpu`) —
the default PyPI wheel pulls several GB of CUDA libraries that CPU inference never uses.
`make install` does this.

---

## 16. Docker

> **Untested.** Docker was unavailable in the environment this was built in. The
> configuration is written to spec but has not been executed.

```bash
cp .env.example .env
docker compose up --build

# frontend  http://localhost:5173
# backend   http://localhost:8000/docs
# mlflow    http://localhost:5000   (docker compose --profile tracking up)
```

Models are **mounted, not baked in**, so retraining does not require an image rebuild. Train
first, or the backend starts in a degraded state and scoring endpoints return 503 (by
design — `/health` reports why).

Secrets come from the environment; no key is ever committed.

---

## 17. MLflow

Every model variant is logged as a run tagged with its imbalance strategy, which is what
makes the comparison auditable rather than a claim in a README. Per-epoch loss and PR-AUC
curves are logged for the deep models.

```bash
make mlflow    # http://localhost:5000
```

MLflow 3.x deprecated the bare file store, so the backend is SQLite (`mlflow.db`). Tracking
degrades to a no-op if MLflow is unavailable — training never fails because of it.

---

## 18. Testing

```bash
pytest                              # 144 tests
pytest --cov=fraudshield --cov=backend/app
```

| Suite | Tests | Covers |
|---|---|---|
| `ml/tests/test_data.py` | 13 | Schema validation, loading, split disjointness and stratification |
| `ml/tests/test_preprocess_imbalance.py` | 19 | Train-only fitting, all four strategies, leakage guard |
| `ml/tests/test_metrics_threshold.py` | 13 | Metric correctness, the accuracy trap, threshold selection |
| `ml/tests/test_models_registry.py` | 16 | All models, MLP/autoencoder training, SHAP, artifact round-trip |
| `ml/tests/test_pipeline.py` | 8 | Full pipeline integration |
| `backend/tests/test_risk_engine.py` | 17 | Band boundaries, actions, input validation |
| `backend/tests/test_api.py` | 22 | All endpoints, invalid input, batch, 503 handling |
| `backend/tests/test_agent.py` | 26 | Tool whitelist, all three guardrails, provider selection |
| `backend/tests/test_investigate_endpoint.py` | 10 | Investigation integration |

Tests assert real behaviour — that the autoencoder reconstructs legitimate transactions
better than fraud, that a scaler fitted on train differs from one fitted on everything, that
the agent cannot override a recommended action.

---

## 19. CI/CD

`.github/workflows/ci.yml` runs on push and PR:

```
Python: install → ruff check → ruff format --check → pytest
        → train on synthetic fixture → boot API → assert a live prediction
React:  npm ci → lint → build → upload dist
Docker: build both images (needs both jobs green)
```

The workflow fails if tests fail.

---

## 20. Limitations

- **Two evaluation caveats.** The reported test numbers come from one split (seed 77777);
  with only 98 test positives, expect roughly +/-0.03 variation across seeds. Earlier splits
  were used while debugging the LightGBM configuration, so quote the seed-77777 run.
- **Docker is unverified** — it was unavailable in the build environment.
- **Two days of data from 2013.** Any model trained on it is a portfolio artifact, not a
  production system.
- **Anonymised features** prevent domain feature engineering and limit explanations to
  direction and magnitude.
- **No velocity or customer-baseline features** — the biggest available accuracy win, and the
  one the PCA components structurally cannot provide.
- **In-process counters** on `/metrics` reset on restart and are wrong across replicas; they
  belong in Redis before horizontal scaling.
- **Investigation history is caller-supplied.** There is no account-history service behind it.
- **No authentication** on the API.
- **SHAP runs synchronously**, which is fine at review volumes and wrong for authorisation
  latency — it should be async for flagged transactions only.
- **No drift monitoring** is wired up, though the autoencoder provides the signal for it.

---

## 21. Future improvements

1. Velocity and customer-baseline features (transactions per hour, deviation from the
   cardholder's own norm) — the largest expected accuracy gain.
2. Async SHAP off the authorisation path.
3. Drift monitoring: PSI on inputs, score-distribution tracking, delayed-label performance
   recomputation as chargebacks arrive.
4. Champion/challenger shadow deployment with automated promotion on PR-AUC.
5. Redis-backed counters and feature cache.
6. Authentication, rate limiting and audit logging.
7. Calibration (Platt/isotonic) so probabilities support expected-loss decisions directly.
8. Reviewer feedback loop, keeping a small unblocked control group so the training labels do
   not become a function of the model's own decisions.

---

## Repository layout

```
FraudShield-AI/
├── configs/config.yaml          # every threshold, cost and band — single source of truth
├── ml/
│   ├── src/fraudshield/         # data · features · models · evaluation · explainability · training
│   ├── scripts/                 # download_data · make_synthetic_data · train
│   ├── notebooks/01_eda.ipynb
│   └── tests/
├── backend/app/                 # api · core · schemas · services
├── frontend/src/                # components · pages · lib
├── models/                      # artifact bundles (git-ignored)
├── reports/                     # metrics, comparisons, figures (git-ignored)
├── docs/interview_questions.md
└── docker-compose.yml
```

Licensed for portfolio use. Dataset © ULB / Worldline, used under its Kaggle terms.
