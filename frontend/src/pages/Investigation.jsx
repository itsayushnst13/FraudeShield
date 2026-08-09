import { useEffect, useState } from 'react';
import PredictionResult from '../components/PredictionResult';
import TransactionForm from '../components/TransactionForm';
import { ErrorNotice, SyntheticNotice } from '../components/Notices';
import { api, emptyTransaction } from '../lib/api';

const SAMPLE = { V14: -4.6, V10: -3.9, V12: -3.4, V17: -3.7, V4: 2.9, Amount: 878.4, Time: 7200 };

export default function Investigation() {
  const [transaction, setTransaction] = useState({
    ...emptyTransaction(), ...SAMPLE, transaction_id: 'txn_2041', customer_id: 'cust_774',
  });
  const [history, setHistory] = useState([{ Amount: 42.1, Time: 1200 }, { Amount: 68.9, Time: 5400 }]);
  const [data, setData] = useState(null);
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState({ busy: false, error: '' });

  useEffect(() => { api.modelInfo().then(setModel).catch(() => setModel(null)); }, []);

  const run = async () => {
    setStatus({ busy: true, error: '' });
    try {
      setData(await api.investigate(transaction, history));
    } catch (err) {
      setData(null);
      setStatus({ busy: false, error: err.message });
      return;
    }
    setStatus({ busy: false, error: '' });
  };

  const updateHistory = (index, field) => (e) => {
    const next = [...history];
    next[index] = { ...next[index], [field]: Number(e.target.value) };
    setHistory(next);
  };

  const report = data?.report;

  return (
    <>
      <div className="page-head">
        <h1>Investigation</h1>
        <p>
          Score a transaction, then generate a written investigation from the evidence. The model
          owns the decision; the report explains it and never re-scores it.
        </p>
      </div>

      <SyntheticNotice show={model?.is_synthetic} dataSource={model?.data_source} />
      <ErrorNotice message={status.error} />

      <div className="card">
        <div className="card-title">Transaction under review</div>
        <TransactionForm value={transaction} onChange={setTransaction} />
      </div>

      <div className="card">
        <div className="card-title">Account history supplied to the agent</div>
        <p style={{ color: 'var(--muted)', marginTop: 0, fontSize: 12.5 }}>
          With no history the agent reports “Insufficient evidence.” rather than inventing context.
        </p>
        {history.map((row, i) => (
          <div className="grid cols-3" key={i} style={{ marginBottom: 8 }}>
            <div><label>Amount</label><input type="number" step="0.01" value={row.Amount} onChange={updateHistory(i, 'Amount')} /></div>
            <div><label>Time</label><input type="number" step="1" value={row.Time} onChange={updateHistory(i, 'Time')} /></div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button className="ghost" onClick={() => setHistory(history.filter((_, j) => j !== i))}>Remove</button>
            </div>
          </div>
        ))}
        <div className="actions">
          <button className="ghost" onClick={() => setHistory([...history, { Amount: 0, Time: 0 }])}>Add transaction</button>
          <button className="ghost" onClick={() => setHistory([])}>Clear history</button>
          <button onClick={run} disabled={status.busy}>
            {status.busy ? <><span className="spinner" /> Investigating</> : 'Run investigation'}
          </button>
        </div>
      </div>

      {data ? (
        <>
          <PredictionResult result={data.prediction} policy={model?.risk_policy} />

          <div className="card report">
            <div className="card-title">
              Investigation report · {report.generated_by === 'llm' ? `${report.llm_provider}/${report.llm_model}` : 'deterministic (no LLM configured)'}
            </div>
            <h4>Risk summary</h4>
            <p>{report.risk_summary}</p>
            <h4>Why it was flagged</h4>
            <p>{report.why_flagged}</p>
            <h4>Key evidence</h4>
            <ul>{report.key_evidence.map((b, i) => <li key={i}>{b}</li>)}</ul>
            <h4>Recommended action</h4>
            <p className="mono">{report.recommended_action}</p>
            <h4>Confidence and limitations</h4>
            <p>{report.confidence_and_limitations}</p>
            {report.warnings?.map((w) => <div className="notice warn" key={w} style={{ marginTop: 12 }}>{w}</div>)}
          </div>

          <div className="card">
            <div className="card-title">Tools the agent was allowed to call</div>
            <table>
              <thead><tr><th>Tool</th><th>Result</th><th>Note</th></tr></thead>
              <tbody>
                {data.evidence.map((e) => (
                  <tr key={e.tool}>
                    <td className="mono">{e.tool}</td>
                    <td style={{ color: e.available ? 'var(--risk-low)' : 'var(--muted)' }}>
                      {e.available ? 'returned data' : 'no data'}
                    </td>
                    <td style={{ color: 'var(--muted)' }}>{e.note || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="card"><p className="empty">Run an investigation to see the report.</p></div>
      )}
    </>
  );
}
