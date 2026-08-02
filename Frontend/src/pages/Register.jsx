import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import { ToastContainer, useToastManager } from '../components/Toast';
import { apiGet, apiPost } from '../api.jsx';

function SampleDots({ count }) {
  return (
    <div style={{ display: 'flex', gap: '.3rem', flexWrap: 'wrap', marginTop: '.5rem', justifyContent: 'center' }}>
      {Array(10).fill(0).map((_, i) => (
        <div key={i} style={{
          width: 14, height: 14, borderRadius: '50%',
          border: '1px solid',
          borderColor: i < count ? 'var(--green)' : 'var(--border)',
          background: i < count ? 'var(--green)' : 'var(--bg3)',
          boxShadow: i < count ? '0 0 5px rgba(57,211,83,.5)' : 'none',
          transition: '.2s',
        }} />
      ))}
    </div>
  );
}

export default function Register() {
  const [isLive, setIsLive]           = useState(false);
  const [name, setName]               = useState('');
  const [rollNumber, setRollNumber]   = useState('');
  const [department, setDepartment]   = useState('');
  const [previewSrc, setPreviewSrc]   = useState(null);
  const [previewError, setPreviewError] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [progress, setProgress]       = useState(0);
  const [samplesEst, setSamplesEst]   = useState(0);
  const [result, setResult]           = useState(null);
  const [statusVisible, setStatusVisible] = useState(false);
  const { toasts, addToast }          = useToastManager();
  const liveIntervalRef               = useRef(null);
  const progIntervalRef               = useRef(null);

  // Poll monitoring status
  useEffect(() => {
    const poll = async () => {
      const data = await apiGet('/api/stats');
      if (data) setIsLive(data.monitoring);
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  // Camera hamesha live chalti rahe — mount pe shuru, unmount pe band
  useEffect(() => {
    startLivePreview();
    return () => stopLivePreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function startLivePreview() {
    if (liveIntervalRef.current) return; // pehle se chal rahi hai
    setPreviewError(false);
    setPreviewSrc(`/api/register/frame?t=${Date.now()}`);
    liveIntervalRef.current = setInterval(() => {
      setPreviewSrc(`/api/register/frame?t=${Date.now()}`);
    }, 500);
  }

  function stopLivePreview() {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
  }

  function refreshPreview() {
    setPreviewError(false);
    // Agar live nahi chal rahi toh shuru karo
    if (!liveIntervalRef.current) {
      startLivePreview();
    }
  }

  async function startRegistration() {
    if (!name.trim()) {
      addToast('❌ Please enter a student name!', 'red');
      return;
    }

    // Camera band NAHI karni — live chalti rahe registration ke dauran bhi
    setRegistering(true);
    setStatusVisible(true);
    setProgress(0);
    setSamplesEst(0);
    setResult(null);

    // Fake progress animation
    let prog = 0;
    progIntervalRef.current = setInterval(() => {
      prog = Math.min(prog + 1.8, 88);
      setProgress(prog);
      setSamplesEst(Math.floor(prog / 10));
    }, 200);

    const r = await apiPost('/api/register', {
      name:        name.trim(),
      roll_number: rollNumber.trim(),
      department:  department.trim(),
    });

    clearInterval(progIntervalRef.current);
    setProgress(100);
    setSamplesEst(r?.captured || 0);

    if (r && r.success) {
      setResult({
        success: true,
        icon: '🎉',
        message: `${name.trim()} registered successfully!`,
        detail: `ID: ${r.student_id} · Roll: ${r.roll_number || '—'} · Dept: ${r.department || '—'} · ${r.captured} samples`,
        sub: 'Student is now ready for face recognition during monitoring.',
        captured: r.captured,
      });
      addToast(`✅ ${name.trim()} registered!`, 'green');
      setName('');
      setRollNumber('');
      setDepartment('');
      // Camera rukti nahi — bas preview refresh
      setPreviewError(false);
    } else {
      setResult({
        success: false,
        icon: '❌',
        message: r?.error || 'Registration failed',
        detail: r?.error || 'Could not capture face. Ensure good lighting and face is visible.',
        sub: 'Adjust lighting and try again.',
        captured: r?.captured || 0,
      });
      addToast('❌ Registration failed!', 'red');
    }
    setRegistering(false);
  }

  return (
    <>
      <ToastContainer toasts={toasts} />
      <Navbar isLive={isLive} />
      <main className="main">

        {/* Register Card */}
        <div className="card">
          <div className="card-title"><span>📷</span> Register New Student</div>
          <div className="reg-grid">

            {/* Left: Camera Preview — hamesha live */}
            <div>
              <div className="reg-preview">
                {previewError || !previewSrc ? (
                  <div style={{ color: 'var(--muted)', textAlign: 'center' }}>
                    <div style={{ fontSize: '2.5rem' }}>📷</div>
                    <p style={{ fontSize: '.85rem' }}>Camera not accessible</p>
                  </div>
                ) : (
                  <img
                    src={previewSrc}
                    alt="Camera Preview"
                    onError={() => setPreviewError(true)}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                )}
                {/* Live indicator */}
                <div style={{
                  position: 'absolute', bottom: '.6rem', left: '.6rem',
                  padding: '.25rem .65rem', borderRadius: '99px', fontSize: '.72rem',
                  fontWeight: 600,
                  background: previewError ? 'rgba(255,77,79,.2)' : 'rgba(57,211,83,.2)',
                  border: `1px solid ${previewError ? 'rgba(255,77,79,.5)' : 'rgba(57,211,83,.5)'}`,
                  color: previewError ? 'var(--red)' : 'var(--green)',
                }}>
                  {previewError ? '❌ No Camera' : registering ? '🔴 Capturing...' : '🟢 Live'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '.5rem', marginTop: '.5rem' }}>
                <button
                  className="btn btn-ghost"
                  style={{ flex: 1, fontSize: '.8rem' }}
                  onClick={refreshPreview}
                >
                  🔄 Refresh
                </button>
              </div>
            </div>

            {/* Right: Form */}
            <div>
              {/* Name */}
              <div className="form-group">
                <label className="form-label">Student Full Name *</label>
                <input
                  className="form-input"
                  placeholder="e.g. Ali Hassan"
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && startRegistration()}
                  disabled={registering}
                />
              </div>

              {/* Roll Number */}
              <div className="form-group">
                <label className="form-label">Roll Number</label>
                <input
                  className="form-input"
                  placeholder="e.g. 22-CS-045"
                  type="text"
                  value={rollNumber}
                  onChange={e => setRollNumber(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && startRegistration()}
                  disabled={registering}
                />
              </div>

              {/* Department */}
              <div className="form-group">
                <label className="form-label">Department</label>
                <input
                  className="form-input"
                  placeholder="e.g. Computer Science"
                  type="text"
                  value={department}
                  onChange={e => setDepartment(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && startRegistration()}
                  disabled={registering}
                />
              </div>

              <button
                className="btn btn-primary"
                style={{ width: '100%', marginTop: '.25rem' }}
                onClick={startRegistration}
                disabled={registering}
              >
                {registering ? '⏳ Capturing... (camera still live)' : '📸 Capture & Register (10 samples)'}
              </button>

              {/* Status Panel */}
              {statusVisible && (
                <div className="reg-status">
                  <div style={{ fontSize: '2.5rem', marginBottom: '.5rem' }}>
                    {result ? result.icon : '⏳'}
                  </div>
                  <div style={{ fontWeight: 600, marginBottom: '.4rem' }}>
                    {result ? result.message : 'Capturing face samples...'}
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                  <div style={{ fontSize: '.8rem', color: 'var(--muted)', marginBottom: '.5rem' }}>
                    {samplesEst} / 10 captured
                  </div>
                  <SampleDots count={samplesEst} />
                </div>
              )}

              {/* Result */}
              {result && (
                <div style={{
                  display: 'block', marginTop: '.75rem', padding: '.75rem',
                  borderRadius: 8,
                  border: `1px solid ${result.success ? 'var(--green)' : 'var(--red)'}`,
                  background: result.success ? 'rgba(57,211,83,.05)' : 'rgba(255,77,79,.05)',
                }}>
                  <div style={{ fontSize: '.85rem' }}>
                    <strong style={{ color: result.success ? 'var(--green)' : 'var(--red)' }}>
                      {result.success ? '✅ Success' : '❌ Failed'}
                    </strong>
                    {' — '}{result.detail}<br />
                    <span style={{ color: 'var(--muted)', fontSize: '.8rem' }}>{result.sub}</span>
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>

        {/* Tips Card */}
        <div className="card">
          <div className="card-title"><span>📋</span> Tips for Best Registration Accuracy</div>
          <div className="tips-grid">
            {[
              { icon: '💡', text: 'Good lighting — face camera directly, avoid backlight' },
              { icon: '😊', text: 'Neutral expression — remove glasses if possible' },
              { icon: '📏', text: 'Keep 50–80cm distance from camera' },
              { icon: '🔁', text: '10 samples captured — more = better recognition' },
              { icon: '🧠', text: 'FaceNet vggface2 — requires ≥90% detection confidence' },
              { icon: '🎯', text: 'Re-register if recognition rate seems low during monitoring' },
            ].map((tip, i) => (
              <div key={i} className="tip-item">
                <div className="tip-icon">{tip.icon}</div>
                <div className="tip-text">{tip.text}</div>
              </div>
            ))}
          </div>
        </div>

      </main>
    </>
  );
}
