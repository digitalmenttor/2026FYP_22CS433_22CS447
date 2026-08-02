import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import { ToastContainer, useToastManager } from '../components/Toast';
import { apiGet, riskColor } from '../api.jsx';

// Generate avatar gradient from name
function strToColor(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = str.charCodeAt(i) + ((h << 5) - h);
  const hue = Math.abs(h) % 360;
  return `linear-gradient(135deg, hsl(${hue},55%,38%), hsl(${(hue + 50) % 360},55%,28%))`;
}

const EV_COLOR_MAP = {
  yaw: 'badge-orange', mobile: 'badge-red', peeking: 'badge-blue',
  paper_exchange: 'badge-orange', book: 'badge-purple',
};

// ── Student Detail Modal ──────────────────────────────────────────────────────
function StudentModal({ studentId, studentName, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!studentId) return;
    setLoading(true);
    apiGet(`/api/students/${studentId}/detail`).then(d => {
      setData(d);
      setLoading(false);
    });
  }, [studentId]);

  return (
    <div
      style={{
        display: 'block', position: 'fixed', inset: 0, zIndex: 300,
        background: 'rgba(0,0,0,.75)', backdropFilter: 'blur(8px)',
        overflowY: 'auto', padding: '2rem',
      }}
      onClick={onClose}
    >
      <div
        style={{
          maxWidth: 780, margin: '0 auto', background: 'var(--bg2)',
          border: '1px solid var(--border)', borderRadius: 16, padding: '1.5rem',
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.2rem' }}>👤 {studentName}</h2>
          <button className="btn btn-ghost" style={{ padding: '.35rem .75rem' }} onClick={onClose}>✕ Close</button>
        </div>

        {loading && <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted)' }}>⏳ Loading...</div>}
        {!loading && (!data || data.error) && (
          <p style={{ color: 'var(--red)' }}>{data?.error || 'Failed to load'}</p>
        )}
        {!loading && data && !data.error && <ModalBody data={data} />}
      </div>
    </div>
  );
}

function ModalBody({ data }) {
  const s = data.student || {};
  const ev = data.cheating_events || [];
  const eng = data.engagement || [];
  const al = data.alert_summary || {};
  const engAvg = eng.length
    ? (eng.reduce((a, e) => a + e.engagement_score, 0) / eng.length).toFixed(1)
    : 'N/A';

  return (
    <>
      {/* Info + Alert Breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div className="card" style={{ margin: 0 }}>
          <div className="card-title"><span>ℹ️</span> Info</div>
          <div style={{ fontSize: '.85rem', display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
            <div><span style={{ color: 'var(--muted)' }}>ID:</span> <strong>{s.id}</strong></div>
            <div><span style={{ color: 'var(--muted)' }}>Roll No:</span> <strong>{s.roll_number || '—'}</strong></div>
            <div><span style={{ color: 'var(--muted)' }}>Department:</span> {s.department || '—'}</div>
            <div><span style={{ color: 'var(--muted)' }}>Registered:</span> {s.registered_at || '-'}</div>
            <div><span style={{ color: 'var(--muted)' }}>Sessions Attended:</span> <strong>{data.sessions_attended || 0}</strong></div>
            <div>
              <span style={{ color: 'var(--muted)' }}>Avg Engagement:</span>{' '}
              <strong style={{ color: engAvg > 70 ? 'var(--green)' : engAvg > 40 ? 'var(--yellow)' : 'var(--red)' }}>
                {engAvg}%
              </strong>
            </div>
            <div>
              <span style={{ color: 'var(--muted)' }}>Total Alerts:</span>{' '}
              <strong style={{ color: 'var(--red)' }}>{s.total_alerts || 0}</strong>
            </div>
          </div>
        </div>

        <div className="card" style={{ margin: 0 }}>
          <div className="card-title"><span>📊</span> Alert Breakdown</div>
          <div style={{ fontSize: '.85rem', display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
            {Object.entries(al).length > 0 ? Object.entries(al).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className={`badge ${EV_COLOR_MAP[k] || 'badge-blue'}`} style={{ padding: '.15rem .5rem' }}>{k}</span>
                <span className="status-pill pill-red">{v}×</span>
              </div>
            )) : <span style={{ color: 'var(--muted)' }}>No alerts recorded</span>}
          </div>
        </div>
      </div>

      {/* Cheating Events */}
      <div className="card" style={{ margin: '0 0 1rem' }}>
        <div className="card-title"><span>🚨</span> Cheating Events (last 50)</div>
        {ev.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.82rem' }}>
              <thead>
                <tr style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '.55rem .5rem' }}>Time</th>
                  <th style={{ textAlign: 'left', padding: '.55rem .5rem' }}>Type</th>
                  <th style={{ textAlign: 'left', padding: '.55rem .5rem' }}>Severity</th>
                  <th style={{ textAlign: 'left', padding: '.55rem .5rem' }}>Direction/Device</th>
                </tr>
              </thead>
              <tbody>
                {ev.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,.05)' }}>
                    <td style={{ color: 'var(--muted)', padding: '.55rem .5rem' }}>{e.occurred_at}</td>
                    <td style={{ padding: '.55rem .5rem' }}>
                      <span className={`badge ${EV_COLOR_MAP[e.event_type] || 'badge-blue'}`}>{e.event_type}</span>
                    </td>
                    <td style={{ padding: '.55rem .5rem' }}>
                      <span className={`status-pill ${e.severity === 'HIGH' ? 'pill-red' : 'pill-yellow'}`}>{e.severity}</span>
                    </td>
                    <td style={{ color: 'var(--muted)', padding: '.55rem .5rem' }}>
                      {e.direction || e.device_class || (e.paper_from ? `P${e.paper_from}→P${e.paper_to}` : '-')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p style={{ color: 'var(--muted)', fontSize: '.85rem' }}>No cheating events recorded ✅</p>}
      </div>

      {/* Engagement History */}
      <div className="card" style={{ margin: 0 }}>
        <div className="card-title"><span>📈</span> Engagement History</div>
        {eng.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            {eng.map((e, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '.75rem', fontSize: '.8rem' }}>
                <span style={{ color: 'var(--muted)', minWidth: 150 }}>{e.recorded_at}</span>
                <div style={{ flex: 1, background: 'var(--bg3)', borderRadius: 99, height: 7, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 99,
                    width: `${e.engagement_score}%`,
                    background: e.engagement_score > 70 ? 'var(--green)' : e.engagement_score > 40 ? 'var(--yellow)' : 'var(--red)',
                  }} />
                </div>
                <span style={{ minWidth: 45, textAlign: 'right', fontWeight: 700 }}>
                  {e.engagement_score?.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        ) : <p style={{ color: 'var(--muted)', fontSize: '.85rem' }}>No engagement data yet</p>}
      </div>
    </>
  );
}

// ── Sessions Table ────────────────────────────────────────────────────────────
function SessionsTable({ sessions }) {
  if (!sessions.length) {
    return <p style={{ color: 'var(--muted)', fontSize: '.85rem' }}>No sessions recorded yet</p>;
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.82rem' }}>
        <thead>
          <tr style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
            {['ID', 'Started', 'Duration', 'Students', 'Yaw', 'Peek', 'Mobile', 'Paper', 'Export'].map(h => (
              <th key={h} style={{ textAlign: h === 'Students' || h === 'Yaw' || h === 'Peek' || h === 'Mobile' || h === 'Paper' ? 'center' : 'left', padding: '.55rem .5rem' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sessions.map(s => (
            <tr key={s.id} style={{ borderBottom: '1px solid rgba(255,255,255,.05)' }}>
              <td style={{ color: 'var(--muted)', padding: '.55rem .5rem' }}>#{s.id}</td>
              <td style={{ padding: '.55rem .5rem' }}>{s.started_at || '-'}</td>
              <td style={{ color: 'var(--muted)', padding: '.55rem .5rem' }}>{s.duration_sec ? s.duration_sec + 's' : '-'}</td>
              <td style={{ textAlign: 'center', padding: '.55rem .5rem' }}>{s.students_present || 0}</td>
              <td style={{ textAlign: 'center', padding: '.55rem .5rem', color: s.total_yaw_alerts > 0 ? 'var(--red)' : 'var(--muted)' }}>{s.total_yaw_alerts || 0}</td>
              <td style={{ textAlign: 'center', padding: '.55rem .5rem', color: s.total_peeking_alerts > 0 ? 'var(--yellow)' : 'var(--muted)' }}>{s.total_peeking_alerts || 0}</td>
              <td style={{ textAlign: 'center', padding: '.55rem .5rem', color: s.total_mobile_alerts > 0 ? 'var(--red)' : 'var(--muted)' }}>{s.total_mobile_alerts || 0}</td>
              <td style={{ textAlign: 'center', padding: '.55rem .5rem', color: s.total_paper_exchanges > 0 ? 'var(--orange)' : 'var(--muted)' }}>{s.total_paper_exchanges || 0}</td>
              <td style={{ padding: '.55rem .5rem' }}>
                <a href={`/api/export_report/${s.id}`} className="btn btn-ghost" style={{ padding: '.3rem .6rem', fontSize: '.75rem' }}>⬇️ CSV</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main Students Page ────────────────────────────────────────────────────────
export default function Students() {
  const [isLive, setIsLive] = useState(false);
  const [students, setStudents] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [riskDetail, setRiskDetail] = useState({});
  const [loading, setLoading] = useState(true);
  const [modalStudent, setModalStudent] = useState(null); // { id, name }
  const { toasts, addToast } = useToastManager();

  useEffect(() => {
    loadAll();
    const interval = setInterval(async () => {
      const data = await apiGet('/api/stats');
      if (data) setIsLive(data.monitoring);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  async function loadAll() {
    setLoading(true);
    const [studentsData, riskData, sessionsData] = await Promise.all([
      apiGet('/api/students'),
      apiGet('/api/risk_scores'),
      apiGet('/api/sessions'),
    ]);
    setStudents(studentsData?.students || []);
    setRiskDetail(riskData?.detail || {});
    setSessions(sessionsData?.sessions || []);
    setLoading(false);
  }

  async function deleteStudent(id, name) {
    if (!window.confirm(`Delete "${name}"? This will remove them from the database.`)) return;
    const r = await fetch(`/api/students/${id}`, { method: 'DELETE' });
    const data = await r.json();
    if (data.success) {
      addToast(`🗑️ ${name} deleted`, 'yellow');
      loadAll();
    } else {
      addToast(`❌ Delete failed: ${data.error}`, 'red');
    }
  }

  return (
    <>
      <ToastContainer toasts={toasts} />
      {modalStudent && (
        <StudentModal
          studentId={modalStudent.id}
          studentName={modalStudent.name}
          onClose={() => setModalStudent(null)}
        />
      )}
      <Navbar isLive={isLive} />
      <main className="main">

        {/* Top bar */}
        <div style={{ display: 'flex', gap: '.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <button className="btn btn-ghost" onClick={loadAll}>🔄 Refresh</button>
          {students.length > 0 && (
            <span style={{ color: 'var(--muted)', fontSize: '.85rem' }}>
              {students.length} student{students.length !== 1 ? 's' : ''} registered
            </span>
          )}
        </div>

        {/* Students List */}
        <div className="card">
          <div className="card-title"><span>👥</span> Registered Students</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            {loading && <p style={{ color: 'var(--muted)' }}>Loading...</p>}
            {!loading && students.length === 0 && (
              <p style={{ color: 'var(--muted)' }}>
                No students registered yet.{' '}
                <a href="/register" style={{ color: 'var(--blue)' }}>Register one →</a>
              </p>
            )}
            {!loading && students.map(s => {
              const risk = riskDetail[s.id] || {};
              const riskScore = risk.risk_score || 0;
              const riskLvl = risk.risk_level || 'GREEN';
              const rc = riskColor(riskLvl);
              const hasAlerts = s.total_alerts > 0;
              return (
                <div key={s.id} className="student-item">
                  <div className="student-avatar" style={{ background: strToColor(s.name) }}>
                    {s.name.charAt(0).toUpperCase()}
                  </div>
                  <div className="student-info">
                    <div className="student-name">{s.name}</div>
                    <div className="student-meta">
                      ID: {s.id}
                      {s.roll_number ? ' · Roll: ' + s.roll_number : ''}
                      {s.department ? ' · ' + s.department : ''}
                      {s.registered_at ? ' · Reg: ' + s.registered_at.split(' ')[0] : ''}
                      {' · Sessions: '}{s.sessions_attended || 0}
                      {' · Alerts: '}
                      <strong style={{ color: hasAlerts ? 'var(--red)' : 'var(--muted)' }}>
                        {s.total_alerts || 0}
                      </strong>
                      {s.avg_engagement ? ` · Eng: ${s.avg_engagement}%` : ''}
                    </div>
                    {riskScore > 0 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', marginTop: '.2rem' }}>
                        <div style={{ width: 70, height: 5, borderRadius: 99, background: 'var(--bg3)', overflow: 'hidden', flexShrink: 0 }}>
                          <div style={{ height: '100%', borderRadius: 99, width: `${Math.min(100, riskScore)}%`, background: rc }} />
                        </div>
                        <span style={{ fontSize: '.7rem', fontWeight: 700, color: rc }}>{riskScore.toFixed(0)} {riskLvl}</span>
                        {(risk.reasons || []).slice(0, 1).map((r, i) => (
                          <span key={i} style={{ fontSize: '.68rem', color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 200, whiteSpace: 'nowrap' }}>{r}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="student-actions">
                    {hasAlerts
                      ? <span className="status-pill pill-red">🚨 {s.total_alerts}</span>
                      : <span className="status-pill pill-green">✅ Clean</span>}
                    <button
                      className="btn-icon view"
                      title="View Detail"
                      onClick={() => setModalStudent({ id: s.id, name: s.name })}
                    >👁️</button>
                    <button
                      className="btn-icon"
                      title="Delete"
                      onClick={() => deleteStudent(s.id, s.name)}
                    >🗑️</button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Sessions */}
        <div className="card">
          <div className="card-title"><span>📅</span> Recent Sessions</div>
          {loading
            ? <p style={{ color: 'var(--muted)', fontSize: '.85rem' }}>Loading...</p>
            : <SessionsTable sessions={sessions} />
          }
        </div>

      </main>
    </>
  );
}
