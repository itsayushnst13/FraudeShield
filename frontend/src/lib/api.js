// Thin API client. Every call surfaces server-provided error detail rather than
// a generic failure message, so a 503 "model not loaded" reads as exactly that.

const BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch {
    throw new Error(`Cannot reach the API at ${BASE}. Is the backend running?`);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    if (Array.isArray(detail)) {
      throw new Error(detail.map((d) => `${d.loc?.slice(1).join('.')}: ${d.msg}`).join('; '));
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

export const api = {
  health: () => request('/health'),
  modelInfo: () => request('/model-info'),
  metrics: () => request('/metrics'),
  predict: (transaction) =>
    request('/predict', { method: 'POST', body: JSON.stringify(transaction) }),
  batchPredict: (transactions) =>
    request('/batch-predict', { method: 'POST', body: JSON.stringify({ transactions }) }),
  investigate: (transaction, history = []) =>
    request('/investigate', { method: 'POST', body: JSON.stringify({ transaction, history }) }),
};

export const RISK_COLORS = {
  LOW: 'var(--risk-low)',
  MEDIUM: 'var(--risk-medium)',
  HIGH: 'var(--risk-high)',
  CRITICAL: 'var(--risk-critical)',
};

export const PCA_FIELDS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);

export function emptyTransaction() {
  const t = { Time: 0, Amount: 0, transaction_id: '', customer_id: '' };
  PCA_FIELDS.forEach((f) => { t[f] = 0; });
  return t;
}

export const fmtPct = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`;

export const fmtNum = (v, digits = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits);
