import { useState, useCallback, useRef } from 'react';

let _addToast = null;

export function useToastManager() {
  const [toasts, setToasts] = useState([]);
  const lastToastTime = useRef({});

  const addToast = useCallback((msg, type) => {
    const now = Date.now();
    if (lastToastTime.current[msg] && now - lastToastTime.current[msg] < 4000) return;
    lastToastTime.current[msg] = now;
    const id = now + Math.random();
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);

  _addToast = addToast;

  return { toasts, addToast };
}

export function showToast(msg, type) {
  if (_addToast) _addToast(msg, type);
}

export function ToastContainer({ toasts }) {
  return (
    <div className="alert-banner">
      {toasts.map(t => (
        <div key={t.id} className={`alert-toast toast-${t.type}`}>
          {t.msg}
        </div>
      ))}
    </div>
  );
}
