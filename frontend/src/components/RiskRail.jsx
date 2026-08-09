import { RISK_COLORS, fmtPct } from '../lib/api';

/**
 * The console's signature element: the 0..1 probability line drawn with the
 * configured risk bands, the model's decision threshold, and where this
 * transaction landed. It makes visible the thing a fraud reviewer most needs
 * and a bare percentage hides — that the cut-off is a tuned business choice,
 * not 0.5.
 */
export default function RiskRail({ probability, threshold, riskLevel, policy = [] }) {
  const bands = policy.length
    ? [...policy].sort((a, b) => a.min_probability - b.min_probability)
    : [
        { risk_level: 'LOW', min_probability: 0 },
        { risk_level: 'MEDIUM', min_probability: 0.3 },
        { risk_level: 'HIGH', min_probability: 0.7 },
        { risk_level: 'CRITICAL', min_probability: 0.9 },
      ];

  const segments = bands.map((band, i) => {
    const start = band.min_probability;
    const end = i + 1 < bands.length ? bands[i + 1].min_probability : 1;
    return { ...band, width: (end - start) * 100 };
  });

  const pct = Math.min(Math.max(probability ?? 0, 0), 1) * 100;

  return (
    <div className="riskrail">
      <div className="track">
        {segments.map((s) => (
          <div
            key={s.risk_level}
            className="seg"
            style={{ width: `${s.width}%`, background: RISK_COLORS[s.risk_level] }}
            title={`${s.risk_level} from ${fmtPct(s.min_probability, 0)}`}
          />
        ))}
        {threshold !== undefined && threshold !== null && (
          <div className="thr" style={{ left: `${threshold * 100}%` }} title={`Decision threshold ${threshold}`} />
        )}
        <div
          className="marker"
          style={{ left: `calc(${pct}% - 1.5px)`, background: RISK_COLORS[riskLevel] || 'var(--ink)' }}
          title={`This transaction: ${fmtPct(probability, 1)}`}
        />
      </div>

      <div className="scale">
        <span>0%</span>
        <span>
          decision threshold {threshold !== undefined ? fmtPct(threshold, 0) : '—'} · this transaction{' '}
          {fmtPct(probability, 1)}
        </span>
        <span>100%</span>
      </div>

      <div className="legend">
        {segments.map((s) => (
          <span key={s.risk_level}>
            <i style={{ background: RISK_COLORS[s.risk_level] }} />
            {s.risk_level} ≥ {fmtPct(s.min_probability, 0)}
            {s.action ? ` · ${s.action.replaceAll('_', ' ').toLowerCase()}` : ''}
          </span>
        ))}
      </div>
    </div>
  );
}
