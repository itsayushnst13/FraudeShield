import { useEffect, useState } from 'react';
import PRCurve from '../components/PRCurve';
import { ErrorNotice, Loading, SyntheticNotice } from '../components/Notices';
import { api, fmtNum, fmtPct } from '../lib/api';

export default function Performance() {
  const [state, setState] = useState({ loading: true });

  useEffect(() => {
    Promise.all([api.metrics(), api.modelInfo()])
      .then(([metrics, model]) => setState({ loading: false, metrics, model }))
      .catch((err) => setState({ loading: false, error: err.message }));
  }, []);

  if (state.loading) return <Loading label="Loading model performance" />;
  if (state.error) {
    return (
      <>
        <div className="page-head"><h1>Model performance</h1></div>
        <ErrorNotice message={state.error} />
      </>
    );
  }

  const { model, metrics } = state;
  const offline = metrics.offline_metrics || {};
  const test = model.test_metrics || {};
  const cm = offline.confusion_matrix;
  const comparison = offline.comparison || [];
  const imbalance = offline.imbalance_study || [];
  const baseline = cm ? (cm[1][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) : undefined;

  return (
    <>
      <div className="page-head">
        <h1>Model performance</h1>
        <p>Held-out test results for the deployed model, and the validation comparison that selected it.</p>
      </div>

      <SyntheticNotice show={model.is_synthetic} dataSource={model.data_source} />

      <div className="grid cols-4">
        {[
          ['PR-AUC', fmtNum(test.pr_auc, 4)],
          ['ROC-AUC', fmtNum(test.roc_auc, 4)],
          ['Precision', fmtPct(test.precision, 1)],
          ['Recall', fmtPct(test.recall, 1)],
        ].map(([label, value]) => (
          <div className="card stat" key={label}>
            <div className="label">{label}</div>
            <div className="value">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="card-title">Precision-recall curve (test)</div>
          <PRCurve curve={offline.pr_curve} baseline={baseline} />
          <p style={{ fontSize: 12, color: 'var(--muted)' }}>
            The dashed line is the fraud base rate — what a model that guesses at random would
            achieve. Distance above it is the real signal.
          </p>
        </div>

        <div className="card">
          <div className="card-title">Confusion matrix at threshold {model.threshold}</div>
          {cm ? (
            <>
              <div className="confusion">
                <div className="hdr" />
                <div className="hdr">predicted legit</div>
                <div className="hdr">predicted fraud</div>
                <div className="hdr">actual legit</div>
                <div className="cell hit"><b>{cm[0][0].toLocaleString()}</b>true negative</div>
                <div className="cell miss"><b>{cm[0][1].toLocaleString()}</b>false alarm</div>
                <div className="hdr">actual fraud</div>
                <div className="cell miss"><b>{cm[1][0].toLocaleString()}</b>missed fraud</div>
                <div className="cell hit"><b>{cm[1][1].toLocaleString()}</b>caught fraud</div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 12 }}>
                F1 {fmtNum(test.f1, 4)} · a missed fraud costs far more than a false alarm, which is
                why the threshold sits at {model.threshold} rather than 0.50.
              </p>
            </>
          ) : (
            <p className="empty">No confusion matrix recorded. Re-run the training pipeline.</p>
          )}
        </div>
      </div>

      {comparison.length > 0 && (
        <div className="card">
          <div className="card-title">Validation comparison — how the production model was chosen</div>
          <table>
            <thead>
              <tr>
                <th>Model</th><th className="num">Threshold</th><th className="num">Precision</th>
                <th className="num">Recall</th><th className="num">F1</th>
                <th className="num">ROC-AUC</th><th className="num">PR-AUC</th><th>Imbalance</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.model} className={row.model === offline.best_model ? 'best' : ''}>
                  <td className="mono">{row.model}</td>
                  <td className="num">{row.threshold}</td>
                  <td className="num">{fmtNum(row.precision, 3)}</td>
                  <td className="num">{fmtNum(row.recall, 3)}</td>
                  <td className="num">{fmtNum(row.f1, 3)}</td>
                  <td className="num">{fmtNum(row.roc_auc, 3)}</td>
                  <td className="num">{fmtNum(row.pr_auc, 3)}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{row.imbalance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {imbalance.length > 0 && (
        <div className="card">
          <div className="card-title">Class imbalance study (XGBoost held fixed)</div>
          <table>
            <thead>
              <tr>
                <th>Strategy</th><th className="num">Train rows</th><th className="num">Train positives</th>
                <th className="num">PR-AUC</th><th className="num">Precision @0.5</th><th className="num">Recall @0.5</th>
              </tr>
            </thead>
            <tbody>
              {imbalance.map((row) => (
                <tr key={row.strategy}>
                  <td className="mono">{row.strategy}</td>
                  <td className="num">{row.train_rows?.toLocaleString()}</td>
                  <td className="num">{row.train_positives?.toLocaleString()}</td>
                  <td className="num">{fmtNum(row.pr_auc, 4)}</td>
                  <td className="num">{fmtNum(row['precision@0.5'], 3)}</td>
                  <td className="num">{fmtNum(row['recall@0.5'], 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
