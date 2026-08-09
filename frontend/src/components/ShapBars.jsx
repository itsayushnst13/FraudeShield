import { fmtNum } from '../lib/api';

/**
 * Diverging bars for SHAP contributions. Bars sit either side of a centre axis
 * because the sign is the point: a feature can push a transaction toward or
 * away from fraud, and collapsing that into magnitude alone would hide the
 * evidence that clears a transaction.
 */
export default function ShapBars({ contributions = [] }) {
  if (!contributions.length) {
    return <p className="empty">No attributions were computed for this prediction.</p>;
  }
  const max = Math.max(...contributions.map((c) => Math.abs(c.shap_value)), 1e-9);

  return (
    <div>
      {contributions.map((c) => {
        const ratio = (Math.abs(c.shap_value) / max) * 50;
        const positive = c.shap_value > 0;
        return (
          <div className="shap-row" key={c.feature}>
            <span className="name">{c.feature}</span>
            <div className="shap-track">
              <div className="axis" />
              <div
                className="bar"
                style={{
                  width: `${ratio}%`,
                  [positive ? 'left' : 'right']: '50%',
                  background: positive ? 'var(--risk-critical)' : 'var(--risk-low)',
                }}
              />
            </div>
            <span className="val">{c.shap_value > 0 ? '+' : ''}{fmtNum(c.shap_value, 4)}</span>
          </div>
        );
      })}
      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 10 }}>
        Bars right of the axis pushed the score toward fraud; bars left pushed away from it.
        V1–V28 are anonymised PCA components, so direction and magnitude are interpretable but
        their business meaning is not.
      </p>
    </div>
  );
}
