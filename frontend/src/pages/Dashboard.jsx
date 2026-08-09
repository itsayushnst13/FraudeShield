import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ErrorNotice, Loading, SyntheticNotice } from '../components/Notices';
import { api, fmtNum, fmtPct } from '../lib/api';

function Stat({ label, value, hint }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export default function Dashboard() {
  const [state, setState] = useState({ loading: true });

  useEffect(() => {
    Promise.all([api.metrics(), api.modelInfo()])
      .then(([metrics, model]) => setState({ loading: false, metrics, model }))
      .catch((err) => setState({ loading: false, error: err.message }));
  }, []);

  if (state.loading) return <Loading label="Loading dashboard" />;
  if (state.error) {
    return (
      <>
        <div className="page-head"><h1>Dashboard</h1></div>
        <ErrorNotice message={state.error} />
        <div className="card">
          <p>
            The dashboard needs a trained model. Run{' '}
            <code>python ml/scripts/train.py</code> to produce one, then reload.
          </p>
        </div>
      </>
    );
  }

  const { metrics, model } = state;
  const test = model.test_metrics || {};

  return (
    <>
      <div className="page-head">
        <h1>Fraud review dashboard</h1>
        <p>
          Live scoring counters for this API process, alongside the held-out test results of the
          model currently deployed.
        </p>
      </div>

      <SyntheticNotice show={model.is_synthetic} dataSource={model.data_source} />

      <div className="grid cols-4">
        <Stat label="Transactions scored" value={metrics.total_transactions_scored.toLocaleString()} hint="Since this process started" />
        <Stat label="Flagged as fraud" value={metrics.fraud_detected.toLocaleString()} hint={`At threshold ${model.threshold}`} />
        <Stat label="Flag rate" value={fmtPct(metrics.fraud_rate, 2)} hint="Share of scored traffic alerted" />
        <Stat label="Deployed model" value={model.model_type} hint={`v${model.version} · ${model.n_features} features`} />
      </div>

      <div className="grid cols-3" style={{ marginTop: 16 }}>
        <Stat label="Test PR-AUC" value={fmtNum(test.pr_auc, 4)} hint="Primary selection metric" />
        <Stat label="Test recall" value={fmtPct(test.recall, 1)} hint="Share of real fraud caught" />
        <Stat label="Test precision" value={fmtPct(test.precision, 1)} hint="Share of alerts that were fraud" />
      </div>

      <div className="card">
        <div className="card-title">Why this model is in production</div>
        <p>{model.selection_rationale || 'No selection rationale recorded.'}</p>
        <h3 style={{ marginTop: 14 }}>How the threshold was chosen</h3>
        <p style={{ color: 'var(--muted)', marginTop: 4 }}>{model.threshold_rationale || '—'}</p>
      </div>

      <div className="card">
        <div className="card-title">Training provenance</div>
        <table>
          <tbody>
            <tr><td>Data source</td><td className="mono">{model.data_source}</td></tr>
            <tr><td>Trained on</td><td className="mono">{model.n_train_rows.toLocaleString()} rows · {model.n_train_fraud.toLocaleString()} fraud</td></tr>
            <tr><td>Imbalance strategy</td><td className="mono">{model.imbalance_strategy}</td></tr>
            <tr><td>Training date</td><td className="mono">{model.training_date}</td></tr>
          </tbody>
        </table>
        <div className="actions">
          <Link to="/analyze"><button>Score a transaction</button></Link>
          <Link to="/performance"><button className="ghost">See full model performance</button></Link>
        </div>
      </div>
    </>
  );
}
