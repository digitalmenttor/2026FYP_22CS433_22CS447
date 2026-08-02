import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar.jsx';
import LogContainer from '../components/LogContainer.jsx';
import { ToastContainer, useToastManager } from '../components/Toast.jsx';
import { apiGet, apiPost, riskColor } from '../api.jsx';

function VBadge({ color, children, show }) {
  if (!show) return null;
  return <div className={`v-badge ${color}`}>{children}</div>;
}

function MonitorStatCard({ icon, value, label, isAlert, id }) {
  return (
    <div className={`monitor-stat-card${isAlert ? ' alert' : ''}`} id={id}>
      <div className="icon">{icon}</div>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

function StudentList({ students }) {
  const entries = Object.entries(students || {});
  if (!entries.length) {
    return <p style={{ color: 'var(--muted)', fontSize: '.82rem' }}>No students registered</p>;
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
      {entries.map(([id, s]) => {
        const risk = s.risk_score || 0;
        const lvl = s.risk_level || 'GREEN';
        const rc = riskColor(lvl);
        let badge = <span className="status-pill pill-green">✅ OK</span>;
        if (s.yaw_alert) badge = <span className="status-pill pill-red">🚨 Yaw</span>;
        else if (s.peeking) badge = <span className="status-pill pill-yellow">👀 Peek</span>;
        else if (s.device_alert) badge = <span className="status-pill pill-red">📱 Device</span>;
        const reasons = (s.risk_reasons || []).slice(0, 1).join('');
        return (
          <div className="student-item" key={id}>
            <div className="student-avatar">{s.name.charAt(0).toUpperCase()}</div>
            <div className="student-info">
              <div className="student-name">{s.name}</div>
              <div className="risk-mini">
                <div className="risk-mini-bar">
                  <div className="risk-mini-fill" style={{ width: `${Math.min(100, risk)}%`, background: rc }} />
                </div>
                <span style={{ color: rc, fontWeight: 700 }}>{Math.round(risk)}</span>
                <span className={`status-pill pill-${lvl.toLowerCase()}`}>{lvl}</span>
              </div>
              {reasons && (
                <div className="student-meta" title={reasons} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {reasons}
                </div>
              )}
            </div>
            {badge}
          </div>
        );
      })}
    </div>
  );
}

export default function Monitor() {
  const [monitoringActive, setMonitoringActive] = useState(false);
  const [btnState, setBtnState] = useState('idle'); // idle | loading
  const [stats, setStats] = useState({});
  const [logs, setLogs] = useState([]);
  const [students, setStudents] = useState({});
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
    const init = async () => {
      const data = await apiGet('/api/stats');
      if (data && data.monitoring) {
        setMonitoringActive(true);
      }
    };
    init();
  }, []);

  useEffect(() => {
    const poll = async () => {
      const data = await apiGet('/api/stats');
      if (!data) return;
      setMonitoringActive(data.monitoring);
      const s = data.stats || {};
      setStats(s);
      setLogs(data.logs || []);
      setStudents(data.students || {});
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

  async function toggleMonitoring() {
    if (!monitoringActive) {
      setBtnState('loading');
      const r = await apiPost('/api/monitoring/start');
      if (r && r.success) {
        setMonitoringActive(true);
        addToast('🎥 Monitoring started!', 'green');
      } else {
        addToast('❌ Failed to start!', 'red');
      }
      setBtnState('idle');
    } else {
      const r = await apiPost('/api/monitoring/stop');
      if (r && r.success) {
        setMonitoringActive(false);
        addToast('⏹️ Monitoring stopped', 'green');
      }
    }
  }

  function exportReport() {
    window.open('/api/export_report', '_blank');
  }

  const s = stats;

  return (
    <>
      <ToastContainer toasts={toasts} />
      <Navbar isLive={monitoringActive} />
      <main className="main">

        {/* Controls */}
        <div className="control-row">
          <button
            className={`btn ${monitoringActive ? 'btn-danger' : 'btn-primary'}`}
            onClick={toggleMonitoring}
            disabled={btnState === 'loading'}
          >
            {btnState === 'loading' ? '⏳ Starting...' : monitoringActive ? '⏹️ Stop Monitoring' : '▶️ Start Monitoring'}
          </button>
          <button className="btn btn-ghost" onClick={exportReport}>📥 Export Report</button>
        </div>

        <div className="monitor-grid">

          {/* LEFT: Video + Stats */}
          <div>
            <div className="video-container">
              {!monitoringActive ? (
                <div className="video-placeholder">
                  <div className="big">🎥</div>
                  <p>Click "Start Monitoring" to begin</p>
                </div>
              ) : (
                <img src={`/api/video_feed?${Date.now()}`} alt="Live Feed" />
              )}
              {monitoringActive && (
                <div className="video-overlay">
                  <div className="v-badge green">🟢 LIVE</div>
                  <VBadge color="yellow" show={s.peeking_count > 0}>👀 PEEKING</VBadge>
                  <VBadge color="red" show={s.yaw_alert_count > 0}>🚨 YAW CHEAT</VBadge>
                  <VBadge color="red" show={s.mobile_count > 0}>📱 DEVICE</VBadge>
                  <VBadge color="orange" show={s.book_count > 0}>📚 BOOK</VBadge>
                </div>
              )}
            </div>

            <div className="monitor-stats-grid">
              <MonitorStatCard icon="👥" value={s.person_count || 0} label="Persons" />
              <MonitorStatCard icon="✅" value={s.recognized_count || 0} label="Recognized" />
              <MonitorStatCard icon="👀" value={s.peeking_count || 0} label="Peeking" isAlert={s.peeking_count > 0} />
              <MonitorStatCard icon="🚨" value={s.yaw_alert_count || 0} label="Yaw Alerts" isAlert={s.yaw_alert_count > 0} />
              <MonitorStatCard icon="📱" value={s.mobile_count || 0} label="Devices" isAlert={s.mobile_count > 0} />
              <MonitorStatCard icon="📄" value={s.paper_exchange_count || 0} label="Paper Exch" isAlert={s.paper_exchange_count > 0} />
              <MonitorStatCard icon="📚" value={s.book_count || 0} label="Books" isAlert={s.book_count > 0} />
              <MonitorStatCard icon="⏱️" value={(s.session_time || 0) + 's'} label="Session" />
            </div>
          </div>

          {/* RIGHT: Side Panel */}
          <div className="side-panel">

            {/* Activity Log */}
            <div className="card">
              <div className="card-title"><span>📝</span> Activity Log</div>
              <LogContainer logs={logs} maxHeight={180} />
            </div>

            {/* Students */}
            <div className="card">
              <div className="card-title"><span>👥</span> Monitored Students</div>
              <StudentList students={students} />
            </div>

          </div>
        </div>

      </main>
    </>
  );
}
