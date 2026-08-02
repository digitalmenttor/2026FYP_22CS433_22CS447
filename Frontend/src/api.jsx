export async function apiGet(path) {
  try {
    const r = await fetch(path);
    return await r.json();
  } catch {
    return null;
  }
}

export async function apiPost(path, body = {}) {
  try {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return await r.json();
  } catch {
    return null;
  }
}

export function getLogClass(log) {
  if (log.includes('🚨') || log.includes('ALERT') || log.includes('CHEAT')) return 'alert';
  if (log.includes('⚠️') || log.includes('peek') || log.includes('PEEK')) return 'warn';
  return 'ok';
}

export function riskColor(lvl) {
  return lvl === 'RED' ? 'var(--red)' : lvl === 'YELLOW' ? 'var(--yellow)' : 'var(--green)';
}
