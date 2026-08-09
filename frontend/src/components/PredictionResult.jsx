import { RISK_COLORS, fmtPct } from '../lib/api';
import RiskRail from './RiskRail';
import ShapBars from './ShapBars';

export default function PredictionResult({ result, policy }) {
  if (!result) return null;
  const fraud = result.prediction === 'FRAUD';
  const color = RISK_COLORS[result.risk_level] || 'var(--ink)';

  return (
    <>
      <div className="card">
        <div className="card-title">Decision</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
          <span className="verdict" style={{ color }}>{result.prediction}</span>
          <span className="badge" style={{ color }}>{result.risk_level}</span>
          <span className="badge" style={{ color: 'var(--ink-2)' }}>
            {result.recommended_action.replaceAll('_', ' ')}
          </span>
          <span className="mono" style={{ marginLeft: 'auto', fontSize: 22, fontWeight: 600 }}>
            {fmtPct(result.fraud_probability, 2)}
          </span>
        </div>

        <div style={{ marginTop: 18 }}>
          <RiskRail
            probability={result.fraud_probability}
            threshold={result.threshold}
            riskLevel={result.risk_level}
            policy={policy}
          />
        </div>

        <p style={{ color: 'var(--muted)', marginTop: 14, fontSize: 12.5 }}>
          Scored by {result.model_name} v{result.model_version} ({result.model_type}). The model
          flags a transaction when its probability reaches {fmtPct(result.threshold, 0)}
          {fraud ? ' — this one did.' : ' — this one did not.'}
        </p>

        {result.warnings?.map((w) => (
          <div className="notice warn" key={w} style={{ marginTop: 12 }}>{w}</div>
        ))}
      </div>

      <div className="card">
        <div className="card-title">Why the model scored it this way</div>
        <ShapBars contributions={result.explanation} />
      </div>
    </>
  );
}
