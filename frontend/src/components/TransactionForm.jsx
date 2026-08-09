import { PCA_FIELDS } from '../lib/api';

/**
 * Entry form for the 30 model inputs. The PCA components are collapsed behind a
 * disclosure because an analyst almost never types them by hand — Amount, Time
 * and the identifiers are what a person actually reasons about.
 */
export default function TransactionForm({ value, onChange, showIds = true }) {
  const set = (field) => (event) => {
    const raw = event.target.value;
    const next = field === 'transaction_id' || field === 'customer_id' ? raw : raw === '' ? '' : Number(raw);
    onChange({ ...value, [field]: next });
  };

  return (
    <>
      <div className="grid cols-4">
        <div>
          <label htmlFor="Amount">Amount</label>
          <input id="Amount" type="number" step="0.01" min="0" value={value.Amount} onChange={set('Amount')} />
        </div>
        <div>
          <label htmlFor="Time">Time (seconds since first record)</label>
          <input id="Time" type="number" step="1" min="0" value={value.Time} onChange={set('Time')} />
        </div>
        {showIds && (
          <>
            <div>
              <label htmlFor="transaction_id">Transaction ID (optional)</label>
              <input id="transaction_id" value={value.transaction_id || ''} onChange={set('transaction_id')} />
            </div>
            <div>
              <label htmlFor="customer_id">Customer ID (optional)</label>
              <input id="customer_id" value={value.customer_id || ''} onChange={set('customer_id')} />
            </div>
          </>
        )}
      </div>

      <details style={{ marginTop: 16 }}>
        <summary style={{ cursor: 'pointer', fontSize: 13, color: 'var(--ink-2)' }}>
          PCA components V1–V28 ({PCA_FIELDS.filter((f) => Number(value[f]) !== 0).length} non-zero)
        </summary>
        <div className="field-grid" style={{ marginTop: 12 }}>
          {PCA_FIELDS.map((field) => (
            <div key={field}>
              <label htmlFor={field}>{field}</label>
              <input id={field} type="number" step="0.0001" value={value[field]} onChange={set(field)} />
            </div>
          ))}
        </div>
      </details>
    </>
  );
}
