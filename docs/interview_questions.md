# Interview questions and answers

Answers for the design decisions in this project. Where a number is required, run the
pipeline on the real Kaggle dataset first — placeholders marked `TBD` must be filled with
measured values, never estimated.

---

## Problem framing

### 1. Why fraud detection?

It is the clearest example of a problem where the standard ML playbook actively misleads
you. The positive class is ~0.17% of the data, the two error types have wildly different
costs, the decision has to be explainable to a human reviewer and eventually a regulator,
and the adversary changes behaviour in response to your model. Any one of those breaks a
naive approach. It also maps onto a real operational workflow — a queue of held
transactions worked by analysts — so the system has to produce evidence, not just a score.

### 2. Why isn't accuracy useful here?

Because a model that predicts "legitimate" for every transaction scores 99.83% accuracy and
catches zero fraud. Accuracy is dominated by the majority class, and with a 578:1 imbalance
the majority class is essentially the whole dataset. A metric that a null model can max out
carries no information about whether the model works. `ml/tests/test_metrics_threshold.py`
contains `test_accuracy_is_misleading_under_extreme_imbalance`, which asserts exactly this.

### 3. Why PR-AUC rather than ROC-AUC?

ROC-AUC plots true-positive rate against false-positive rate, and the false-positive rate
divides by the number of true negatives. With 284,315 negatives, thousands of false alarms
barely move that denominator, so ROC-AUC stays high even for a model an operations team
could not use. Precision-recall ignores true negatives entirely: precision asks "of the
alerts we raised, how many were real fraud", which is the question the review queue cares
about. PR-AUC is also threshold-free, so it ranks candidate models without entangling the
comparison with the operating point — which is chosen separately, afterwards.

### 4. What's the cost asymmetry, and how does it enter the model?

A missed fraud costs the chargeback, the write-off and the investigation. A false alarm
costs a few minutes of analyst time plus some customer friction. In `configs/config.yaml`
these are `cost_false_negative: 500` and `cost_false_positive: 5` — a 100:1 ratio. They
enter through threshold selection, which minimises `500 × FN + 5 × FP`, not through the
training loss. Keeping them in config means risk appetite can be retuned without retraining.

---

## Modelling

### 5. Why XGBoost?

Tabular data with non-linear, overlapping class boundaries and no spatial or sequential
structure — the regime where gradient-boosted trees still beat neural networks. Concretely:
it handles unscaled inputs, captures feature interactions without manual crosses, trains in
seconds on 284k rows, exposes `scale_pos_weight` for imbalance, and supports exact TreeSHAP,
which lets explanations run inline in the request path rather than as a batch job.

Note the honest caveat: the project does **not** assume XGBoost wins. `run_pipeline` selects
whichever candidate has the best validation PR-AUC and records the rationale in
`metadata.json`. On the synthetic smoke fixture, random forest was selected.

### 6. Why also build a PyTorch MLP?

Partly to test the assumption rather than assert it — the comparison table is only
meaningful if a neural network was genuinely trained. Partly because the deployment path
differs: an MLP gives a differentiable model that can be fine-tuned incrementally on new
fraud patterns, whereas boosted trees are retrained from scratch. Architecture is
Linear → BatchNorm → ReLU → Dropout blocks into a single logit, trained with
`BCEWithLogitsLoss(pos_weight=...)`.

The important detail: **early stopping monitors validation PR-AUC, not validation loss.**
Loss is dominated by the majority class and keeps improving while fraud recall degrades.
Stopping on loss would select the wrong epoch.

### 7. Why an autoencoder?

It answers a failure mode the supervised models cannot. XGBoost only recognises fraud that
resembles the 492 labelled positives it trained on; when attackers change tactics it fails
silently, still returning confident low scores. The autoencoder trains **only on legitimate
transactions**, learns what normal traffic looks like, and scores by reconstruction error —
so it can flag a novel pattern it has never seen labelled.

The EDA supports this: fraudulent transactions average far more extreme (|z| > 3) PCA
components than legitimate ones. But the overlap is substantial, so it runs as a novelty
and drift tripwire beside the supervised scorer, not as the primary decision-maker. It is
explicitly excluded from production model selection, because its output is a rank score
rather than a calibrated probability.

### 8. Why SMOTE — and did you use it?

SMOTE synthesises minority examples by interpolating between a positive and its nearest
positive neighbours, giving the model more of the decision boundary to learn from than
duplication would. It is compared in the imbalance study, with XGBoost held fixed so the
strategy is the only variable.

The default is `class_weight`, not SMOTE. Reweighting the loss achieves a similar effect
without inventing rows, and synthetic positives interpolated in a 30-dimensional PCA space
have no guarantee of corresponding to a plausible transaction. SMOTE also carries a real
leakage hazard: applied before splitting, a synthetic point in training can be an
interpolation of a point that ends up in validation. `apply_strategy` therefore only ever
receives the training fold.

### 9. How did you avoid data leakage?

Four structural controls rather than good intentions:

1. **Split first.** `make_splits` runs before any fitting. Test is sealed until the final
   evaluation and opened exactly once.
2. **Preprocessor fitted on train only.** `fit_preprocessor(X_train)` — the scaler never
   sees validation or test statistics. Tested by
   `test_scaler_is_fitted_on_training_data_only`.
3. **Resampling on the training fold only**, with `assert_no_resampling_on_eval` available
   as an explicit tripwire that compares class distributions.
4. **Raw `Time` dropped.** `Time` is seconds since the dataset's first record — it encodes
   position in this specific collection window, which does not exist for future traffic. A
   cyclical hour-of-day encoding replaces it.

Model selection and threshold selection both happen on validation, so test contaminates no
decision.

### 10. How did you choose the threshold?

By sweeping a grid of candidate operating points on the **validation** split and minimising
expected cost, subject to a minimum-precision guardrail that keeps alert volume within what
a review team can absorb. `select_threshold` records the winning point, the full sweep table
and a written rationale, all persisted to `metadata.json` and surfaced at `GET /model-info`.

0.5 is the right threshold only when classes are balanced and errors are symmetric — neither
is true here. The threshold is also configurable, because the correct value is a business
decision that changes with fraud losses and staffing.

---

## Explainability and the GenAI layer

### 11. Why SHAP?

Because an analyst holding a transaction needs to know why *this* one scored 0.94, not which
features matter on average. SHAP attributes the gap between the model's base rate and this
prediction across features, with contributions that sum to the gap. That is the shape of
evidence a reviewer needs, and it is also what an adverse-action explanation requires if the
decision is ever challenged.

TreeSHAP is exact for tree ensembles and fast enough to run inside the request. Non-tree
models fall back to a sampling explainer over a stored background sample.

The honest limitation: V1–V28 are anonymised PCA components. The system can say "V14 pushed
this score up by 0.42" but never "the merchant category was suspicious", and the prompt
forbids the LLM from claiming otherwise.

### 12. Why use an LLM *after* the ML model rather than instead of it?

Different jobs. The ML model is a calibrated, testable, versioned scorer whose behaviour is
measurable on a held-out set. An LLM is a fluent writer with no reliable probability
estimate over tabular features, no calibration, and no way to be regression-tested against a
PR curve. Asking it to score fraud would replace a measurable component with an unmeasurable
one.

What the LLM is good at is the part the ML model can't do: turning a probability, a SHAP
vector and an account history into something a human can act on in fifteen seconds.

### 13. Why shouldn't the LLM make the fraud decision — and how is that enforced?

Three enforcement layers, because a prompt instruction alone is not a control:

1. **Prompt.** The system prompt states the ML score is authoritative and forbids inventing
   evidence, mandating the exact phrase "Insufficient evidence." when a tool returns nothing.
2. **Validation.** `_validate_narrative` rejects any narrative containing phrases that
   contradict the ML verdict, and the deterministic report is used instead.
3. **Architecture.** The authoritative fields — probability, risk band, recommended action —
   are copied from the model and risk engine into the response and are *never parsed back
   out of LLM text*. Even a narrative that slipped through validation could not change the
   decision, only the prose beside it.

`backend/tests/test_agent.py` tests all three, including
`test_llm_cannot_override_the_recommended_action`.

### 14. How does the agent work?

`Transaction → ML model → probability → SHAP → toolkit → agent → report.`

The agent gets a whitelisted, read-only toolkit: `get_transaction_details`,
`get_customer_transaction_history`, `get_model_prediction`, `get_shap_explanation`,
`get_risk_policy`, `get_model_info`. No database handle, no shell, no filesystem. Calls to
anything outside the whitelist are blocked and logged.

Every tool returns a `ToolResult` with an explicit `available` flag, which is what lets the
agent say "Insufficient evidence." truthfully instead of fabricating account history. If no
LLM is configured, or the response is malformed, or validation fails, the system falls back
to a deterministic report assembled from the evidence — so an investigation is always
produced. The provider is swappable via `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_MODEL`.

---

## Production and operations

### 15. How would you deploy this at scale?

Split the paths. Real-time authorisation needs a p99 under ~100ms: serve the model behind a
horizontally-scaled service with the artifact baked into the image, warm at startup, and
keep SHAP off the synchronous path — compute it asynchronously for transactions that get
flagged, since 99.8% of traffic is cleared and never reviewed.

Bulk scoring goes through a separate batch path. Put a queue between scoring and
investigation so LLM latency never blocks an authorisation. Cache aggressively on
customer-level features. The service is already stateless apart from in-process counters,
which would move to Redis or a metrics backend before running multiple replicas.

### 16. How would you monitor model drift?

Three layers, because they fail at different speeds:

- **Input drift** (fast, no labels needed): population stability index or KS tests on each
  feature's distribution against the training reference, per day. Also monitor the score
  distribution itself — a shifting mean predicted probability is an early warning.
- **Output drift**: alert rate and risk-band mix. A sudden change with no deployment
  attached usually means the input feed changed.
- **Performance drift** (slow, needs labels): chargebacks arrive weeks late, so maintain a
  delayed-label pipeline and recompute PR-AUC and recall on each cohort as labels land.

The autoencoder's reconstruction error is a fourth signal: a rising share of traffic above
the calibrated threshold suggests traffic no longer resembles the training distribution.

### 17. What happens when fraud patterns change?

Assume they will — this is adversarial, not stationary. Concretely: the supervised model
degrades quietly while its inputs still look reasonable, which is why the autoencoder runs
alongside it. Practically, you want a scheduled retrain on a rolling window with the
threshold re-optimised each time (the optimal operating point moves with the base rate), a
champion/challenger setup so a new model shadows production before taking traffic, and a
fast path to ship a rules-based patch while a retrain runs — rules are worse models but
deploy in minutes.

### 18. How would you reduce false positives?

In rough order of return:

1. **Move the threshold.** The precision/recall curve is already measured; this is a config
   change and needs no retraining.
2. **Add features the current model lacks** — velocity (transactions per hour), device and
   IP reputation, merchant history, geo-distance from the last transaction. The PCA
   components are a heavily constrained view of a transaction.
3. **Customer-level baselining.** A £500 transaction is unremarkable for one cardholder and
   extreme for another; the model currently has no per-customer context.
4. **Tiered actions.** Not every alert needs a block — step-up authentication for the
   HIGH band converts a hard decline into a small friction, which is what the `risk_engine`
   action mapping is for.
5. **Feed reviewer outcomes back** as labels so the model learns from confirmed false alarms.

### 19. How would you handle millions of transactions?

Throughput is not the hard part — a boosted tree scores in microseconds and the service
scales horizontally. The real constraints are: feature computation (velocity features need a
low-latency store like Redis, not a database join), the review queue (at 1% alert rate, a
million transactions produce 10,000 alerts — the threshold has to be set against analyst
headcount, which is why `min_precision` exists), and cost control on the LLM layer, which
should only ever run on the flagged minority.

### 20. How would you retrain the model?

Scheduled and automated, with the same pipeline that produced the current artifact — that
is precisely why the pipeline is a module rather than a notebook. New model gets a new
version in `metadata.json`, is compared against the incumbent on a common held-out set, and
ships only if it wins on PR-AUC with acceptable recall. MLflow keeps the run history so any
regression can be traced to the parameters that caused it. Rollback is swapping the model
directory, since the artifact bundles the preprocessor with the estimator.

The failure mode to watch: retraining on data whose labels came from the model's own
decisions. Blocked transactions never generate a chargeback, so they look "legitimate" if
you are careless — a feedback loop that quietly teaches the model to stop catching what it
already catches. Keeping a small random control group unblocked is the standard remedy.
