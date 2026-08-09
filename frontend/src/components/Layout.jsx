import { NavLink } from 'react-router-dom';

const LINKS = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/analyze', label: 'Transaction analyzer' },
  { to: '/investigate', label: 'Investigation' },
  { to: '/performance', label: 'Model performance' },
];

export default function Layout({ children, health }) {
  const online = health?.model_loaded;
  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <strong>FraudShield</strong>
          <span>AI</span>
        </div>
        <nav className="nav">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end}>
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="rail-foot">
          <span className={`dot ${online ? 'ok' : 'bad'}`} />
          {online ? 'Model ready' : 'No model loaded'}
          {health?.api_version ? <div>API v{health.api_version}</div> : null}
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
