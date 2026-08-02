import { getLogClass } from '../api.jsx';

export default function LogContainer({ logs, maxHeight = 220 }) {
  if (!logs || logs.length === 0) {
    return (
      <div className="log-container" style={{ maxHeight }}>
        <div className="log-entry ok">System ready — waiting for monitoring to start</div>
      </div>
    );
  }

  return (
    <div className="log-container" style={{ maxHeight }}>
      {[...logs].reverse().map((log, i) => (
        <div key={i} className={`log-entry ${getLogClass(log)}`}>{log}</div>
      ))}
    </div>
  );
}
