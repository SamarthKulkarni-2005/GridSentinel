import React, { useState, useEffect, useCallback } from 'react';

const STATUS_LABELS = {
  critical: 'CRITICAL', priority: 'PRIORITY', high: 'HIGH', moderate: 'MODERATE', normal: 'NORMAL',
};

function riskColor(score) {
  if (score >= 80) return 'var(--crimson)';
  if (score >= 65) return '#ff6d00';
  if (score >= 50) return 'var(--amber)';
  if (score >= 35) return '#ffd54f';
  return 'var(--emerald)';
}

function ShapModal({ meterId, shapData, onClose }) {
  if (!shapData) return null;
  const max = Math.max(...shapData.map(f => Math.abs(f.contribution)), 0.01);

  return (
    <div className="shap-modal-overlay" onClick={onClose}>
      <div className="shap-modal" onClick={e => e.stopPropagation()}>
        <div className="shap-modal-hdr">
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--cyan)' }}>
            SHAP FEATURES — {meterId}
          </span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: '1.1rem' }}>✕</button>
        </div>
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {shapData.slice(0, 7).map((f, i) => {
            const pct = Math.abs(f.contribution) / max * 100;
            return (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-2)' }}>
                    {f.feature.replace(/_/g, ' ')}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--cyan)' }}>
                    {f.contribution.toFixed(4)}
                  </span>
                </div>
                <div style={{ height: 6, background: 'var(--bg-card-3)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    width: `${pct}%`, height: '100%',
                    background: f.contribution >= 0 ? 'var(--crimson)' : 'var(--emerald)',
                    borderRadius: 3, transition: 'width 0.4s ease',
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function AlertView({ apiHeaders }) {
  const [clusters, setClusters]     = useState([]);
  const [loading, setLoading]       = useState(true);
  const [filter, setFilter]         = useState('ALL');
  const [search, setSearch]         = useState('');
  const [sortKey, setSortKey]       = useState('risk');
  const [sortDir, setSortDir]       = useState('desc');
  const [shapData, setShapData]     = useState(null);
  const [shapMeter, setShapMeter]   = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  const fetchClusters = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/anomaly/clusters?top_n=50', { headers: apiHeaders });
      const data = await res.json();
      setClusters(data);
    } catch {
      setClusters([]);
    } finally {
      setLoading(false);
    }
  }, [apiHeaders]);

  useEffect(() => { fetchClusters(); }, [fetchClusters]);

  const openShap = async (meterId) => {
    setShapMeter(meterId);
    setShapLoading(true);
    setShapData(null);
    try {
      const res = await fetch(`/api/shap/${meterId}`, { headers: apiHeaders });
      const data = await res.json();
      setShapData(data);
    } catch {
      setShapData([]);
    } finally {
      setShapLoading(false);
    }
  };

  const FILTER_OPTIONS = ['ALL', 'CRITICAL', 'HIGH', 'MODERATE', 'NORMAL'];
  const filterMap = { CRITICAL: ['critical', 'priority'], HIGH: ['high'], MODERATE: ['moderate'], NORMAL: ['normal'] };

  const counts = {
    critical:  clusters.filter(c => c.status === 'critical').length,
    priority:  clusters.filter(c => c.status === 'priority').length,
    high:      clusters.filter(c => c.status === 'high').length,
    moderate:  clusters.filter(c => c.status === 'moderate').length,
    normal:    clusters.filter(c => c.status === 'normal').length,
  };

  let displayed = [...clusters];
  if (filter !== 'ALL') displayed = displayed.filter(c => filterMap[filter]?.includes(c.status));
  if (search.trim()) displayed = displayed.filter(c => c.feeder.toLowerCase().includes(search.toLowerCase()));

  displayed.sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  return (
    <div className="full-page-view">
      {/* Category counts */}
      <div className="alert-counts">
        {[
          { label: 'CRITICAL', count: counts.critical + counts.priority, color: 'var(--crimson)' },
          { label: 'HIGH',     count: counts.high,                        color: 'var(--amber)' },
          { label: 'MODERATE', count: counts.moderate,                    color: '#ffd54f' },
          { label: 'NORMAL',   count: counts.normal,                      color: 'var(--emerald)' },
        ].map(item => (
          <div key={item.label} className="alert-count-chip" style={{ borderColor: item.color + '44', background: item.color + '12' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.2rem', fontWeight: 700, color: item.color }}>{item.count}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)', letterSpacing: '0.1em' }}>{item.label}</span>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div className="alert-filter-bar">
        <div style={{ display: 'flex', gap: 6 }}>
          {FILTER_OPTIONS.map(f => (
            <button key={f}
              className={`filter-btn ${filter === f ? 'active' : ''}`}
              onClick={() => setFilter(f)}
            >{f}</button>
          ))}
        </div>
        <input
          className="chat-input"
          style={{ width: 220, padding: '6px 12px', fontSize: '0.75rem' }}
          placeholder="Search meter ID..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <div className="skeleton-block" style={{ height: 200, margin: 16 }} />
        ) : (
          <table className="anomaly-table" style={{ fontSize: '0.8rem' }}>
            <thead>
              <tr>
                <th onClick={() => toggleSort('feeder')} style={{ cursor: 'pointer' }}>
                  Meter ID {sortKey === 'feeder' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
                </th>
                <th onClick={() => toggleSort('risk')} style={{ cursor: 'pointer' }}>
                  Risk Score {sortKey === 'risk' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
                </th>
                <th onClick={() => toggleSort('status')} style={{ cursor: 'pointer' }}>
                  Status {sortKey === 'status' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
                </th>
                <th>Detection Logic</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {displayed.map((c, i) => (
                <tr key={i} onClick={() => openShap(c.feeder)} style={{ cursor: 'pointer' }}>
                  <td><span className="feeder-id">{c.feeder}</span></td>
                  <td>
                    <span className="risk-score" style={{ color: riskColor(c.risk) }}>
                      {Math.round(c.risk)}
                    </span>
                    <span style={{ color: 'var(--text-3)', fontSize: '0.65rem', marginLeft: 4 }}>/100</span>
                  </td>
                  <td><span className={`status-pill ${c.status}`}>{STATUS_LABELS[c.status] ?? c.status.toUpperCase()}</span></td>
                  <td><span className="logic-chip">{c.logic}</span></td>
                  <td>
                    <button className="action-btn" onClick={e => { e.stopPropagation(); openShap(c.feeder); }}>
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* SHAP modal */}
      {shapMeter && (
        <ShapModal
          meterId={shapMeter}
          shapData={shapLoading ? null : shapData}
          onClose={() => { setShapMeter(null); setShapData(null); }}
        />
      )}
    </div>
  );
}
