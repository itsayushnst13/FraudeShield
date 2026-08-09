/** Shared banners. Errors say what happened and what to do, never just "oops". */

export function ErrorNotice({ message }) {
  if (!message) return null;
  return <div className="notice error">{message}</div>;
}

export function SyntheticNotice({ show, dataSource }) {
  if (!show) return null;
  return (
    <div className="notice warn">
      <strong>Synthetic model.</strong> This model was trained on generated smoke-test data
      {dataSource ? ` (${dataSource})` : ''}, not the real transaction dataset. Scores and metrics
      shown here are for checking the pipeline works — they are not measures of fraud detection
      quality. Train on the Kaggle dataset to get real numbers.
    </div>
  );
}

export function Loading({ label = 'Loading' }) {
  return (
    <p className="empty">
      <span className="spinner" /> {label}…
    </p>
  );
}
