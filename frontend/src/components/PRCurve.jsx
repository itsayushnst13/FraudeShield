/** Hand-drawn SVG precision-recall curve. No chart library: one path, one axis
 *  pair, and full control over the type and colour tokens. */
export default function PRCurve({ curve, baseline }) {
  if (!curve?.recall?.length) return <p className="empty">No curve data available.</p>;

  const W = 420, H = 260, PAD = 38;
  const x = (r) => PAD + r * (W - PAD - 12);
  const y = (p) => H - PAD - p * (H - PAD - 12);

  const points = curve.recall
    .map((r, i) => [x(r), y(curve.precision[i])])
    .sort((a, b) => a[0] - b[0]);
  const path = points.map(([px, py], i) => `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Precision-recall curve">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={PAD} x2={W - 12} y1={y(t)} y2={y(t)} stroke="var(--rule)" strokeWidth="1" />
          <text x={PAD - 7} y={y(t) + 3.5} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="var(--mono)">
            {t.toFixed(2)}
          </text>
          <text x={x(t)} y={H - PAD + 14} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="var(--mono)">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {baseline !== undefined && (
        <line x1={PAD} x2={W - 12} y1={y(baseline)} y2={y(baseline)}
          stroke="var(--risk-critical)" strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
      )}
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
      <line x1={PAD} x2={PAD} y1={12} y2={H - PAD} stroke="var(--rule-strong)" />
      <line x1={PAD} x2={W - 12} y1={H - PAD} y2={H - PAD} stroke="var(--rule-strong)" />
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--muted)">Recall</text>
      <text x={11} y={H / 2} textAnchor="middle" fontSize="10" fill="var(--muted)" transform={`rotate(-90 11 ${H / 2})`}>
        Precision
      </text>
    </svg>
  );
}
