import { useEffect, useState } from 'react';
import PredictionResult from '../components/PredictionResult';
import TransactionForm from '../components/TransactionForm';
import { ErrorNotice, SyntheticNotice } from '../components/Notices';
import { api, emptyTransaction } from '../lib/api';

// A profile that exercises the components the model weighs most heavily, so the
// form is useful without hand-typing 28 numbers. It is an input template only —
// the score always comes from the model.
const HIGH_RISK_SAMPLE = { V14: -4.2, V10: -3.6, V12: -3.1, V17: -3.4, V4: 2.6, Amount: 412.9, Time: 43200 };

export default function Analyzer() {
  const [transaction, setTransaction] = useState(emptyTransaction());
  const [result, setResult] = useState(null);
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState({ busy: false, error: '' });

  useEffect(() => {
    api.modelInfo().then(setModel).catch(() => setModel(null));
  }, []);

  const score = async () => {
    setStatus({ busy: true, error: '' });
    try {
      setResult(await api.predict(transaction));
    } catch (err) {
      setResult(null);
      setStatus({ busy: false, error: err.message });
      return;
    }
    setStatus({ busy: false, error: '' });
  };

  const loadSample = () => {
    setTransaction({ ...emptyTransaction(), ...HIGH_RISK_SAMPLE, transaction_id: 'txn_sample_01', customer_id: 'cust_001' });
    setResult(null);
  };

  const reset = () => {
    setTransaction(emptyTransaction());
    setResult(null);
    setStatus({ busy: false, error: '' });
  };

  return (
    <>
      <div className="page-head">
        <h1>Transaction analyzer</h1>
        <p>Score a single transaction and see which features drove the result.</p>
      </div>

      <SyntheticNotice show={model?.is_synthetic} dataSource={model?.data_source} />
      <ErrorNotice message={status.error} />

      <div className="card">
        <div className="card-title">Transaction</div>
        <TransactionForm value={transaction} onChange={setTransaction} />
        <div className="actions">
          <button onClick={score} disabled={status.busy}>
            {status.busy ? <><span className="spinner" /> Scoring</> : 'Score transaction'}
          </button>
          <button className="ghost" onClick={loadSample} disabled={status.busy}>Fill high-risk example</button>
          <button className="ghost" onClick={reset} disabled={status.busy}>Clear</button>
        </div>
      </div>

      {result ? (
        <PredictionResult result={result} policy={model?.risk_policy} />
      ) : (
        <div className="card"><p className="empty">Enter a transaction and score it to see the result.</p></div>
      )}
    </>
  );
}
