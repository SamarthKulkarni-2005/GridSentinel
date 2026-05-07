import React, { useState, useEffect, useCallback, useRef } from 'react';

function typeColor(targetType) {
  if (!targetType) return 'var(--text-3)';
  if (targetType === 'meter')       return 'var(--cyan)';
  if (targetType === 'transformer') return 'var(--amber)';
  return 'var(--text-2)';
}

function formatTs(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString('en-IN', { hour12: false }).replace(',', '');
  } catch {
    return ts;
  }
}

function actionIcon(action) {
  if (!action) return '·';
  if (action.includes('login'))   return '🔑';
  if (action.includes('anomaly')) return '⚠';
  if (action.includes('demand'))  return '◈';
  if (action.includes('report'))  return '📄';
  if (action.includes('chat'))    return '💬';
  return '·';
}

export default function AuditView() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');
  const intervalRef = useRef(null);

  const fetchLog = useCallback(async () => {
    const token = localStorage.getItem('gs_token');
    if (!token) {
      setError('No session token found. Please log in again.');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/audit/log', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.status === 401) {
        setError('Session expired. Please log in again.');
        setEntries([]);
        return;
      }
      if (!res.ok) {
        setError(`Server error (${res.status}).`);
        return;
      }
      const data = await res.json();
      setEntries(Array.isArray(data) ? data : []);
    } catch {
      setError('Failed to reach the API server.');
    } finally {
      setLoading(false);
    }
  }, []); // stable — reads token inside

  useEffect(() => {
    fetchLog();
    intervalRef.current = setInterval(fetchLog, 30000);
    return () => clearInterval(intervalRef.current);
  }, [fetchLog]);

  return (
    <div className="full-page-view">
      <div className="panel-hdr" style={{ flexShrink: 0 }}>
        <h3>☰ Audit Log — Last 50 Actions</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)' }}>
            Auto-refreshes every 30s
          </span>
          <button
            onClick={fetchLog}
            style={{
              background: 'var(--bg-card-2)', border: '1px solid var(--border)',
              color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
              padding: '4px 10px', borderRadius: 'var(--r-sm)', cursor: 'pointer',
            }}
          >
            Refresh
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(8)].map((_, i) => (
              <div key={i} className="skeleton-block" style={{ height: 36, borderRadius: 4 }} />
            ))}
          </div>
        ) : error ? (
          <div style={{ padding: 32, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--crimson)' }}>
            {error}
          </div>
        ) : entries.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-3)', lineHeight: 1.8 }}>
            <div style={{ fontSize: '1.4rem', marginBottom: 12 }}>☰</div>
            No audit entries yet.<br />
            <span style={{ fontSize: '0.7rem' }}>Navigate to a meter or transformer view to generate log entries.</span>
          </div>
        ) : (
          <table className="anomaly-table" style={{ fontSize: '0.78rem' }}>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Action</th>
                <th>Target</th>
                <th>Type</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => {
                const tc = typeColor(e.target_type);
                return (
                  <tr key={i}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-3)', whiteSpace: 'nowrap' }}>
                      {formatTs(e.timestamp)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-1)' }}>
                      <span style={{ marginRight: 6 }}>{actionIcon(e.action)}</span>
                      {e.action}
                    </td>
                    <td>
                      {e.target_id
                        ? <span style={{ fontFamily: 'var(--font-mono)', color: tc, fontWeight: 600 }}>{e.target_id}</span>
                        : <span style={{ color: 'var(--text-3)' }}>—</span>}
                    </td>
                    <td>
                      {e.target_type ? (
                        <span style={{
                          fontFamily: 'var(--font-mono)', fontSize: '0.65rem', fontWeight: 700,
                          padding: '2px 8px', borderRadius: 'var(--r-sm)',
                          color: tc, background: tc + '18', border: `1px solid ${tc}44`,
                          textTransform: 'uppercase', letterSpacing: '0.06em',
                        }}>
                          {e.target_type}
                        </span>
                      ) : <span style={{ color: 'var(--text-3)' }}>—</span>}
                    </td>
                    <td style={{ fontSize: '0.7rem', color: 'var(--text-2)', maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {e.details || '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
