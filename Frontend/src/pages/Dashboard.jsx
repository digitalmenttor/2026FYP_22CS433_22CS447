import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar.jsx';
import LogContainer from '../components/LogContainer.jsx';
import { ToastContainer, useToastManager } from '../components/Toast.jsx';
import { apiGet } from '../api.jsx';

function ModelStatus({ status }) {
  const models = [
    { id: 'facenet', label: 'FaceNet', sub: 'vggface2', ok: status.facenet },
    { id: 'facedet', label: 'Face Det', sub: 'OpenCV DNN', ok: status.face_det },
    { id: 'paper', label: 'Paper Model', sub: 'best.pt', ok: status.paper_model },
    { id: 'device', label: 'Device Model', sub: 'best1.pt', ok: status.device_model },
    { id: 'gpu', label: 'Compute', sub: status.gpu ? 'CUDA GPU' : 'CPU', ok: status.gpu },
  ];
  return (
    <div className="model-grid">
      {models.map(m => (
        <div key={m.id} className="model-item">
          <div className={`model-dot ${m.ok ? 'ok' : 'fail'}`} />
          <div>
            <div className="model-name">{m.label}</div>
            <div className="model-sub">{m.sub}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function StatCard({ icon, value, label, isAlert }) {
  return (
    <div className={`stat-card${isAlert ? ' alert' : ''}`}>
      <div className="icon">{icon}</div>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

function RiskTable({ scores }) {
  if (!scores || !scores.length) {
    return <p style={{ color: 'var(--muted)', fontSize: '.85rem' }}>No students being monitored</p>;
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="risk-table">
        <thead>
          <tr>
            <th>Student</th>
            <th>Level</th>
            <th>Score</th>
            <th>Reasons</th>
          </tr>
        </thead>
        <tbody>
          {scores.map((s, i) => {
            const lvl = s.risk_level || 'GREEN';
            const score = s.risk_score || 0;
            const barColor = lvl === 'RED' ? 'var(--red)' : lvl === 'YELLOW' ? 'var(--yellow)' : 'var(--green)';
            const reasons = (s.reasons || []).join(' · ') || '';
            return (
              <tr key={i}>
                <td style={{ fontWeight: 600 }}>{s.student_id || '-'}</td>
                <td>
                  <span className={`risk-pill risk-${lvl.toLowerCase()}`}>{lvl}</span>
                </td>
                <td>
                  <div className="risk-bar-wrap">
                    <div className="risk-bar">
                      <div className="risk-bar-fill" style={{ width: `${score}%`, background: barColor }} />
                    </div>
                    <span style={{ minWidth: 32, fontSize: '.8rem', fontWeight: 700 }}>
                      {Math.round(score)}
                    </span>
                  </div>
                </td>
                <td style={{ color: 'var(--muted)', fontSize: '.72rem', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={reasons}>
                  {reasons || '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Dashboard() {
  const [modelStatus, setModelStatus] = useState({});
  const [stats, setStats] = useState({});
  const [logs, setLogs] = useState([]);
  const [riskSummary, setRiskSummary] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const { toasts, addToast } = useToastManager();
  const lastToastRef = useRef({});

  function maybeToast(condition, msg, type) {
    if (!condition) return;
    const now = Date.now();
    if (lastToastRef.current[msg] && now - lastToastRef.current[msg] < 4000) return;
    lastToastRef.current[msg] = now;
    addToast(msg, type);
  }

  useEffect(() => {
    apiGet('/api/status').then(d => d && setModelStatus(d));
  }, []);

  useEffect(() => {
    const poll = async () => {
      const data = await apiGet('/api/stats');
      if (!data) return;
      setIsLive(data.monitoring);
      const s = data.stats || {};
      setStats(s);
      setLogs(data.logs || []);
      setRiskSummary(data.risk_summary || []);
      maybeToast(s.yaw_alert_count > 0, '🚨 Yaw Cheating Detected!', 'red');
      maybeToast(s.peeking_count > 0, '👀 Peeking Detected!', 'yellow');
      maybeToast(s.mobile_count > 0, '📱 Prohibited Device Detected!', 'red');
      maybeToast(s.paper_exchange_count > 0, '📄 Paper Exchange Detected!', 'red');
      maybeToast(s.book_count > 0, '📚 Book/Notes Detected!', 'yellow');
    };
    poll();
    const interval = setInterval(poll, 1500);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const s = stats;

  return (
    <>
      <ToastContainer toasts={toasts} />
      <Navbar isLive={isLive} />
      <main className="main">

        {/* Hero */}
        <div className="hero">
          <h1>Advanced Cheating Detection System</h1>
          <p>FaceNet · YOLOv8 · MediaPipe · Risk Scoring Engine</p>
          <div className="hero-badges">
            <span className="badge badge-blue">🧠 FaceNet Recognition</span>
            <span className="badge badge-green">👁️ Peeking Detection</span>
            <span className="badge badge-orange">📱 Device Detection</span>
            <span className="badge badge-purple">🦴 MediaPipe Pose</span>
            <span className="badge badge-yellow">⚠️ Risk Scoring</span>
          </div>
        </div>

        {/* Model Status */}
        <div className="card">
          <div className="card-title"><span>🤖</span> Model Status</div>
          <ModelStatus status={modelStatus} />
        </div>

        {/* Live Stats */}
        <div className="stats-grid">
          <StatCard icon="👥" value={s.person_count || 0} label="Detected" />
          <StatCard icon="✅" value={s.recognized_count || 0} label="Recognized" />
          <StatCard icon="✋" value={s.hands_detected || 0} label="Hands" />
          <StatCard icon="👀" value={s.peeking_count || 0} label="Peeking" isAlert={s.peeking_count > 0} />
          <StatCard icon="🚨" value={s.yaw_alert_count || 0} label="Yaw Alerts" isAlert={s.yaw_alert_count > 0} />
          <StatCard icon="📄" value={s.paper_exchange_count || 0} label="Paper Exch" isAlert={s.paper_exchange_count > 0} />
          <StatCard icon="📱" value={s.mobile_count || 0} label="Devices Now" isAlert={s.mobile_count > 0} />
          <StatCard icon="📚" value={s.book_count || 0} label="Books Now" isAlert={s.book_count > 0} />
          <StatCard icon="⏱️" value={(s.session_time || 0) + 's'} label="Session" />
        </div>

        {/* Risk Scores */}
        <div className="card">
          <div className="card-title"><span>⚠️</span> Student Risk Scores</div>
          <RiskTable scores={riskSummary} />
        </div>

        {/* Quick Actions */}
        <div className="card">
          <div className="card-title"><span>⚡</span> Quick Actions</div>
          <div className="control-row">
            <a className="btn btn-primary" href="/monitor">🎥 Start Monitoring</a>
            <a className="btn btn-ghost" href="/register">📷 Register Student</a>
            <a className="btn btn-ghost" href="/students">👥 View Students</a>
            <a className="btn btn-ghost" href="/api/export_report" target="_blank" rel="noreferrer">📥 Export Report</a>
          </div>
        </div>

        {/* Activity Log */}
        <div className="card">
          <div className="card-title"><span>📝</span> Recent Activity</div>
          <LogContainer logs={logs} maxHeight={220} />
        </div>

      </main>
    </>
  );
}
