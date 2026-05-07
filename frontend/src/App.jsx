import React, { useState, useRef, useEffect, useCallback } from 'react';
import Login from './components/login';
import TheftChart from './components/TheftChart';
import AlertView from './components/AlertView';
import PredView from './components/PredView';
import AuditView from './components/AuditView';
import axios from 'axios';
import {
  AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';
import './App.css';

// ── Axios instance with auth token interceptor ────────────────────────────────
const api = axios.create({ baseURL: '/api' });
api.interceptors.request.use(cfg => {
  const token = localStorage.getItem('gs_token');
  if (token) cfg.headers['Authorization'] = `Bearer ${token}`;
  return cfg;
});

// Public axios instance (no auth header) for health/public endpoints
const API = '/api';

/* ═══════════════════════════════════════════════════════════
   FALLBACK STATIC DATA (shown while API loads)
═══════════════════════════════════════════════════════════ */
const FALLBACK_DEMAND = [
  { h: '00', actual: 510, pred: 528, q5: 449, q95: 571 }, { h: '01', actual: 478, pred: 491, q5: 421, q95: 536 },
  { h: '02', actual: 455, pred: 469, q5: 400, q95: 510 }, { h: '03', actual: 441, pred: 455, q5: 388, q95: 494 },
  { h: '04', actual: 448, pred: 460, q5: 394, q95: 502 }, { h: '05', actual: 490, pred: 505, q5: 431, q95: 549 },
  { h: '06', actual: 598, pred: 615, q5: 526, q95: 670 }, { h: '07', actual: 742, pred: 760, q5: 653, q95: 831 },
  { h: '08', actual: 880, pred: 902, q5: 774, q95: 986 }, { h: '09', actual: 940, pred: 965, q5: 827, q95: 1053 },
  { h: '10', actual: 982, pred: 1005, q5: 864, q95: 1100 }, { h: '11', actual: 1010, pred: 1038, q5: 889, q95: 1131 },
  { h: '12', actual: 1045, pred: 1072, q5: 920, q95: 1170 }, { h: '13', actual: 1020, pred: 1048, q5: 898, q95: 1142 },
  { h: '14', actual: 995, pred: 1022, q5: 876, q95: 1114 }, { h: '15', actual: 1030, pred: 1058, q5: 906, q95: 1154 },
  { h: '16', actual: 1088, pred: 1115, q5: 957, q95: 1219 }, { h: '17', actual: 1140, pred: 1172, q5: 1003, q95: 1277 },
  { h: '18', actual: 1195, pred: 1228, q5: 1052, q95: 1338 }, { h: '19', actual: 1210, pred: 1248, q5: 1065, q95: 1355 },
  { h: '20', actual: 1185, pred: 1220, q5: 1043, q95: 1327 }, { h: '21', actual: 1090, pred: 1118, q5: 959, q95: 1221 },
  { h: '22', actual: 920, pred: 945, q5: 810, q95: 1030 }, { h: '23', actual: 720, pred: 740, q5: 634, q95: 806 },
];

const FALLBACK_CLUSTERS = [
  { feeder: 'Loading...', risk: 0, logic: '—', status: 'normal' },
];

const STATUS_LABELS = {
  critical: 'CRITICAL', priority: 'PRIORITY', high: 'HIGH', moderate: 'MODERATE', normal: 'NORMAL'
};

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
function riskColor(score) {
  if (score < 50) return 'var(--emerald)';
  if (score <= 75) return 'var(--amber)';
  return 'var(--crimson)';
}
function riskBand(score) {
  if (score < 50) return 'STABLE';
  if (score <= 75) return 'STRESSED';
  return 'CRITICAL';
}
function formatInrLakhs(val) {
  const lakhs = val / 100000;
  return `₹${lakhs.toFixed(1)}L`;
}

/* ═══════════════════════════════════════════════════════════
   SVG GAUGE — Segmented Arc
═══════════════════════════════════════════════════════════ */
function GsiGauge({ score = 68 }) {
  const cx = 110, cy = 100, outerR = 88, innerR = 62;
  const SEGS = 32, SPAN = 240, START = 150;

  function pt(r, deg) {
    const rad = deg * Math.PI / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }
  function segPath(startDeg, endDeg) {
    const o1 = pt(outerR, startDeg), o2 = pt(outerR, endDeg);
    const i1 = pt(innerR, startDeg), i2 = pt(innerR, endDeg);
    const lg = (endDeg - startDeg) > 180 ? 1 : 0;
    return [`M${o1.x},${o1.y}`, `A${outerR},${outerR},0,${lg},1,${o2.x},${o2.y}`,
      `L${i2.x},${i2.y}`, `A${innerR},${innerR},0,${lg},0,${i1.x},${i1.y}`, 'Z'].join('');
  }

  const segAngle = SPAN / SEGS;
  const GAP = 1.8;
  const filled = Math.round((score / 100) * SEGS);
  const color = riskColor(score);

  const segments = Array.from({ length: SEGS }, (_, i) => {
    const s = START + i * segAngle;
    const e = s + segAngle - GAP;
    const isLit = i < filled;
    const segColor = i < Math.round(0.50 * SEGS) ? 'var(--emerald)'
      : i < Math.round(0.75 * SEGS) ? 'var(--amber)' : 'var(--crimson)';
    return { path: segPath(s, e), isLit, segColor };
  });

  return (
    <svg className="gauge-svg" viewBox="0 0 220 160">
      {segments.map((s, i) => (
        <path key={i} d={s.path} fill={s.isLit ? s.segColor : '#1e1e1e'} />
      ))}
      {segments.filter(s => s.isLit).map((s, i) => (
        <path key={'g' + i} d={s.path} fill={s.segColor} opacity="0.18" style={{ filter: 'blur(3px)' }} />
      ))}
      <text x={cx} y={cy - 8} textAnchor="middle" fill={color}
        fontFamily="'JetBrains Mono',monospace" fontSize="28" fontWeight="700">{score}</text>
      <text x={cx} y={cy + 12} textAnchor="middle" fill="#5a5a5a"
        fontFamily="'JetBrains Mono',monospace" fontSize="8" letterSpacing="2">GSI SCORE</text>
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════
   KORAMANGALA MAP
═══════════════════════════════════════════════════════════ */
const DT_POSITIONS = {
  DT_1:  { x: 145, y: 105, r: 6 },
  DT_2:  { x: 195, y: 155, r: 5 },
  DT_3:  { x: 98,  y: 175, r: 5 },
  DT_4:  { x: 240, y: 100, r: 4 },
  DT_5:  { x: 170, y: 225, r: 5 },
  DT_6:  { x: 270, y: 170, r: 4 },
  DT_7:  { x: 60,  y: 120, r: 4 },
};

function KoraMap() {
  const [zones, setZones] = useState({});

  useEffect(() => {
    const token = localStorage.getItem('gs_token');
    fetch('/api/map/zones', token ? { headers: { Authorization: `Bearer ${token}` } } : {})
      .then(r => r.ok ? r.json() : [])
      .then(data => {
        const map = {};
        data.forEach(z => { map[z.dt_id] = z; });
        setZones(map);
      })
      .catch(() => {});
    const t = setInterval(() => {
      const tk = localStorage.getItem('gs_token');
      fetch('/api/map/zones', tk ? { headers: { Authorization: `Bearer ${tk}` } } : {})
        .then(r => r.ok ? r.json() : [])
        .then(data => {
          const map = {};
          data.forEach(z => { map[z.dt_id] = z; });
          setZones(map);
        }).catch(() => {});
    }, 30000);
    return () => clearInterval(t);
  }, []);

  const riskC = { critical: 'var(--crimson)', high: 'var(--amber)', moderate: '#ffd54f', normal: 'var(--emerald)' };

  return (
    <svg viewBox="0 0 320 280" style={{ width: '100%', height: '100%', display: 'block' }}>
      <rect width="320" height="280" fill="#0d0d0d" />
      {[60, 110, 160, 210, 260].map(x => (
        <line key={'v' + x} x1={x} y1="20" x2={x} y2="260" stroke="#1a1a1a" strokeWidth="6" />
      ))}
      {[50, 100, 150, 200, 250].map(y => (
        <line key={'h' + y} x1="20" y1={y} x2="300" y2={y} stroke="#1a1a1a" strokeWidth="6" />
      ))}
      <line x1="60" y1="50" x2="160" y2="150" stroke="#181818" strokeWidth="4" />
      <line x1="160" y1="100" x2="260" y2="200" stroke="#181818" strokeWidth="4" />
      <text x="22" y="38" fill="#252525" fontFamily="'JetBrains Mono',monospace" fontSize="7">KORAMANGALA</text>
      <text x="200" y="38" fill="#252525" fontFamily="'JetBrains Mono',monospace" fontSize="7">HSR LAYOUT</text>
      <text x="22" y="270" fill="#252525" fontFamily="'JetBrains Mono',monospace" fontSize="7">INDIRANAGAR</text>

      {Object.entries(DT_POSITIONS).map(([dtId, pos]) => {
        const zone  = zones[dtId];
        const risk  = zone?.status ?? 'normal';
        const c     = riskC[risk] ?? riskC.normal;
        const isCrit = risk === 'critical' || risk === 'high';
        const gsi   = zone?.gsi_score;
        const crit  = zone?.critical_meters ?? 0;
        return (
          <g key={dtId}>
            {isCrit && <>
              <circle cx={pos.x} cy={pos.y} r={pos.r} fill="none" stroke={c} strokeWidth="1" opacity="0.6">
                <animate attributeName="r" from={pos.r} to={pos.r + 14} dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.6" to="0" dur="2s" repeatCount="indefinite" />
              </circle>
              <circle cx={pos.x} cy={pos.y} r={pos.r} fill="none" stroke={c} strokeWidth="1" opacity="0.4">
                <animate attributeName="r" from={pos.r} to={pos.r + 14} dur="2s" begin="0.7s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.4" to="0" dur="2s" begin="0.7s" repeatCount="indefinite" />
              </circle>
            </>}
            <circle cx={pos.x} cy={pos.y} r={pos.r + 3} fill={c} opacity="0.12" />
            <circle cx={pos.x} cy={pos.y} r={pos.r} fill={c} opacity="0.9" />
            <text x={pos.x + pos.r + 4} y={pos.y - 2}
              fill={c} fontFamily="'JetBrains Mono',monospace" fontSize="6.5" fontWeight="700">
              {dtId}
            </text>
            {gsi != null && (
              <text x={pos.x + pos.r + 4} y={pos.y + 7}
                fill="rgba(255,255,255,0.35)" fontFamily="'JetBrains Mono',monospace" fontSize="5.5">
                GSI {gsi}
              </text>
            )}
            {crit > 0 && (
              <text x={pos.x + pos.r + 4} y={pos.y + 14}
                fill="var(--crimson)" fontFamily="'JetBrains Mono',monospace" fontSize="5">
                {crit} CRIT
              </text>
            )}
          </g>
        );
      })}

      <rect x="12" y="248" width="7" height="7" rx="1" fill="var(--crimson)" opacity="0.85" />
      <text x="23" y="255" fill="#444" fontFamily="'JetBrains Mono',monospace" fontSize="6">CRITICAL</text>
      <rect x="68" y="248" width="7" height="7" rx="1" fill="var(--amber)" opacity="0.85" />
      <text x="79" y="255" fill="#444" fontFamily="'JetBrains Mono',monospace" fontSize="6">HIGH</text>
      <rect x="108" y="248" width="7" height="7" rx="1" fill="#ffd54f" opacity="0.85" />
      <text x="119" y="255" fill="#444" fontFamily="'JetBrains Mono',monospace" fontSize="6">MODERATE</text>
      <rect x="175" y="248" width="7" height="7" rx="1" fill="var(--emerald)" opacity="0.85" />
      <text x="186" y="255" fill="#444" fontFamily="'JetBrains Mono',monospace" fontSize="6">NORMAL</text>
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════
   CUSTOM CHART TOOLTIP
═══════════════════════════════════════════════════════════ */
const ForecastTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#111', border: '1px solid #2a2a2a', borderRadius: 6,
      padding: '8px 12px', fontFamily: "'JetBrains Mono',monospace", fontSize: '0.72rem'
    }}>
      <div style={{ color: '#5a5a5a', marginBottom: 6 }}>{label}:00</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{p.value} kWh</strong>
        </div>
      ))}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════
   SHAP PANEL (inline, below TheftChart)
═══════════════════════════════════════════════════════════ */
function ShapPanel({ shapData }) {
  if (!shapData || shapData.length === 0) return null;
  const top5 = shapData.slice(0, 5);
  const max = Math.max(...top5.map(f => Math.abs(f.contribution)), 0.01);

  return (
    <div style={{
      borderTop: '1px solid var(--border)', padding: '12px 16px',
      background: 'rgba(0,229,255,0.02)', flexShrink: 0,
    }}>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: '0.6rem', fontWeight: 700,
        letterSpacing: '0.12em', textTransform: 'uppercase',
        color: 'var(--cyan)', marginBottom: 10,
      }}>
        SHAP Feature Contributions
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {top5.map((f, i) => {
          const pct = Math.abs(f.contribution) / max * 100;
          return (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--text-2)' }}>
                  {f.feature.replace(/_/g, ' ')}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--cyan)' }}>
                  {f.contribution.toFixed(4)}
                </span>
              </div>
              <div style={{ height: 5, background: 'var(--bg-card-3)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%',
                  background: f.contribution >= 0 ? 'var(--crimson)' : 'var(--emerald)',
                  borderRadius: 3,
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════ */
export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('gs_token'));
  const [activeNav, setActiveNav]             = useState('dashboard');
  const [chatOpen, setChatOpen]               = useState(false);
  const [messages, setMessages]               = useState([
    { role: 'bot', content: 'GridSentinel Online. Monitoring BESCOM connections. Query a meter ID or transformer ID.' }
  ]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [panelData, setPanelData]     = useState(null);
  const [aiInsight, setAiInsight]     = useState('');
  const [gsiScore, setGsiScore]       = useState(68);
  const [darkMode, setDarkMode]       = useState(() => localStorage.getItem('gs_theme') !== 'light');
  const [economicKpi, setEconomicKpi] = useState(null);
  const messagesEndRef  = useRef(null);
  const chatInputRef    = useRef(null);

  const [activeView, setActiveView]           = useState('demand');
  const [meterData, setMeterData]             = useState(null);
  const [shapData, setShapData]               = useState(null);
  const [demandData, setDemandData]           = useState(FALLBACK_DEMAND);
  const [demandLoading, setDemandLoading]     = useState(false);
  const [anomalyClusters, setAnomalyClusters] = useState(FALLBACK_CLUSTERS);
  const [clustersLoading, setClustersLoading] = useState(false);

  // ── Auth headers for sub-components ──────────────────────────────────────────
  const token = localStorage.getItem('gs_token');
  const apiHeaders = token ? { Authorization: `Bearer ${token}` } : {};

  // ── Dark/light mode ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (darkMode) {
      document.body.removeAttribute('data-theme');
    } else {
      document.body.setAttribute('data-theme', 'light');
    }
    localStorage.setItem('gs_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // ── "/" keyboard shortcut → open chat and focus input ────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
        e.preventDefault();
        setChatOpen(true);
        setTimeout(() => chatInputRef.current?.focus(), 100);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // ── Fetch anomaly clusters and initial demand on mount ────────────────────────
  useEffect(() => {
    setClustersLoading(true);
    api.get('/anomaly/clusters')
      .then(r => { if (r.data?.length) setAnomalyClusters(r.data); })
      .catch(() => setAnomalyClusters(FALLBACK_CLUSTERS))
      .finally(() => setClustersLoading(false));

    setDemandLoading(true);
    api.get('/demand/DT_1')
      .then(r => {
        if (r.data?.demand_series?.length) setDemandData(r.data.demand_series);
        if (r.data?.gsi_score != null)     setGsiScore(Math.round(r.data.gsi_score));
      })
      .catch(() => {})
      .finally(() => setDemandLoading(false));

    // Economic KPI
    api.get('/economic/summary')
      .then(r => setEconomicKpi(r.data))
      .catch(() => {});
  }, []);

  // ── SSE connection for live score updates ─────────────────────────────────────
  useEffect(() => {
    if (!isAuthenticated) return;
    let es;
    try {
      es = new EventSource('/api/stream/scores');
      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.gsi != null) setGsiScore(payload.gsi);
          if (payload.scores?.length) {
            setAnomalyClusters(prev => {
              const updated = [...prev];
              payload.scores.forEach(s => {
                const idx = updated.findIndex(c => c.feeder === s.meter_id);
                if (idx >= 0) {
                  updated[idx] = { ...updated[idx], risk: s.risk, status: s.status };
                }
              });
              return updated;
            });
          }
        } catch {}
      };
      es.onerror = () => { es.close(); };
    } catch {}
    return () => { if (es) es.close(); };
  }, [isAuthenticated]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Handle anomaly row click ──────────────────────────────────────────────────
  const handleMeterClick = async (id) => {
    setShapData(null);
    try {
      const r = await api.get(`/anomaly/${id}`);
      setMeterData(r.data);
      setActiveView('theft');
      // Also fetch SHAP
      try {
        const sr = await api.get(`/shap/${id}`);
        setShapData(sr.data);
      } catch {}
    } catch {
      setMessages(p => [...p, { role: 'bot', content: `Could not load anomaly data for ${id}.` }]);
    }
  };

  // ── Handle chat send ──────────────────────────────────────────────────────────
  const handleSend = useCallback(async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setMessages(p => [...p, { role: 'user', content: userMsg }]);
    setInput('');
    setLoading(true);
    try {
      const res = await api.post('/chat', { message: userMsg });
      const botText = res.data.response;
      setMessages(p => [...p, { role: 'bot', content: botText }]);
      setAiInsight(botText);

      if (res.data.trigger_xai && res.data.target_id) {
        const type = res.data.target_type || 'meter';
        if (type === 'transformer') {
          const d = await api.get(`/demand/${res.data.target_id}`);
          setPanelData(d.data);
          if (d.data.gsi_score != null) setGsiScore(Math.round(d.data.gsi_score));
          if (d.data.demand_series?.length) setDemandData(d.data.demand_series);
        } else {
          setShapData(null);
          const d = await api.get(`/anomaly/${res.data.target_id}`);
          setMeterData(d.data);
          setActiveView('theft');
          try {
            const sr = await api.get(`/shap/${res.data.target_id}`);
            setShapData(sr.data);
          } catch {}
        }
      }
    } catch {
      setMessages(p => [...p, { role: 'bot', content: 'Connection error. Is the backend running?' }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const handleLogout = () => {
    localStorage.removeItem('gs_token');
    setIsAuthenticated(false);
  };

  const [reportLoading, setReportLoading] = useState(false);
  const handleDownloadReport = async () => {
    setReportLoading(true);
    try {
      const token = localStorage.getItem('gs_token');
      const resp = await fetch('/api/report/summary', {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `GridSentinel_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Report generation failed: ' + err.message);
    } finally {
      setReportLoading(false);
    }
  };

  // Auto-logout on any 401 from axios (expired token)
  useEffect(() => {
    const id = api.interceptors.response.use(
      r => r,
      err => {
        if (err.response?.status === 401) handleLogout();
        return Promise.reject(err);
      }
    );
    return () => api.interceptors.response.eject(id);
  }, []);

  if (!isAuthenticated) {
    return <Login onLogin={(t) => { setIsAuthenticated(true); }} />;
  }

  const navItems = [
    { id: 'dashboard', icon: '⊞', label: 'GRID' },
    { id: 'anomaly',   icon: '⚠',  label: 'ALERT' },
    { id: 'demand',    icon: '◈',  label: 'PRED' },
    { id: 'audit',     icon: '☰',  label: 'AUDIT' },
  ];

  const currentGsi   = panelData?.gsi_score != null ? Math.round(panelData.gsi_score) : gsiScore;
  const color        = riskColor(currentGsi);
  const band         = riskBand(currentGsi);
  const criticalCount = anomalyClusters.filter(c => c.status === 'critical' || c.status === 'priority').length;

  return (
    <div className="app-shell">

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">&#9889;</div>
        <nav className="sidebar-nav">
          {navItems.map(n => (
            <button key={n.id} className={`nav-btn ${activeNav === n.id ? 'active' : ''}`}
              onClick={() => setActiveNav(n.id)} title={n.id}>
              <span style={{ fontSize: 18 }}>{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
        </nav>
        <button className={`sidebar-chat-btn ${chatOpen ? 'open' : ''}`}
          onClick={() => setChatOpen(o => !o)} title="Query Console (press /)">
          &#128172;
        </button>
        <button
          onClick={handleLogout}
          title="Log out"
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'var(--text-3)', padding: '10px 0', marginBottom: 8,
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            gap: 3, fontSize: 16, width: '100%',
            transition: 'color var(--t)',
          }}
          onMouseEnter={e => e.currentTarget.style.color = 'var(--crimson)'}
          onMouseLeave={e => e.currentTarget.style.color = 'var(--text-3)'}
        >
          <span>⏻</span>
          <span style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' }}>OUT</span>
        </button>
      </aside>

      {/* ── Main Canvas ─────────────────────────────────────── */}
      <div className="main-canvas">

        {/* Top Bar */}
        <header className="topbar">
          <div>
            <div className="topbar-title">GridSentinel OS</div>
            <div className="topbar-sub">BESCOM · Node Alpha · Predictive Demand &amp; Loss Intelligence</div>
          </div>
          <div className="topbar-spacer" />
          <div className="topbar-kpi">
            <div className="status-dot" />
            <div>
              <div className="topbar-kpi-val">1.1CR</div>
              <div className="topbar-kpi-lbl">Connections</div>
            </div>
          </div>
          <div className="topbar-kpi">
            <div>
              <div className="topbar-kpi-val" style={{ color: 'var(--amber)' }}>{criticalCount}</div>
              <div className="topbar-kpi-lbl">Active Alerts</div>
            </div>
          </div>
          <div className="topbar-kpi">
            <div>
              <div className="topbar-kpi-val" style={{ color }}>{currentGsi}%</div>
              <div className="topbar-kpi-lbl">Grid Stress</div>
            </div>
          </div>
          <div className="topbar-kpi">
            <div>
              <div className="topbar-kpi-val">
                {demandData.length ? `${Math.max(...demandData.map(d => d.actual || 0))} kWh` : '—'}
              </div>
              <div className="topbar-kpi-lbl">Peak Load</div>
            </div>
          </div>
          {economicKpi && (
            <div className="topbar-kpi">
              <div>
                <div className="topbar-kpi-val" style={{ color: 'var(--emerald)' }}>
                  {formatInrLakhs(economicKpi.estimated_protection_inr)}
                </div>
                <div className="topbar-kpi-lbl">Protected</div>
              </div>
            </div>
          )}
          {/* Download PDF Report */}
          <button
            className="theme-toggle-btn"
            onClick={handleDownloadReport}
            disabled={reportLoading}
            title="Download Intelligence Report (PDF)"
            style={{ fontSize: '0.75rem', opacity: reportLoading ? 0.6 : 1, width: 'auto', padding: '0 10px', gap: 4, cursor: reportLoading ? 'not-allowed' : 'pointer' }}
          >
            {reportLoading ? '⏳' : '⬇ PDF'}
          </button>
          {/* Dark/Light toggle */}
          <button
            className="theme-toggle-btn"
            onClick={() => setDarkMode(d => !d)}
            title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {darkMode ? '☀' : '◑'}
          </button>
        </header>

        {/* ── Conditional view based on activeNav ───────── */}
        {activeNav === 'dashboard' && (
          <div className="dashboard-grid">

            {/* ── GAUGE ─────────────────────────────────────── */}
            <div className="area-gauge panel">
              <div className="panel-hdr">
                <h3>&#9992; Grid Stress Index</h3>
                <span className="gauge-band-chip"
                  style={{ color, background: `${color}18`, border: `1px solid ${color}44` }}>
                  {band}
                </span>
              </div>
              <div className="gauge-wrap">
                <GsiGauge score={currentGsi} />
                {panelData?.transformer_id && (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.68rem', color: '#5a5a5a', letterSpacing: '0.08em' }}>
                      ACTIVE ASSET
                    </div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.85rem', fontWeight: 700, color: 'var(--violet)', marginTop: 4 }}>
                      {panelData.transformer_id}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-2)', marginTop: 4 }}>
                      Directive: <strong style={{ color: 'var(--crimson)' }}>{panelData.action}</strong>
                    </div>
                  </div>
                )}
                {panelData?.stress_factors && (
                  <div style={{ width: '100%', fontSize: '0.72rem' }}>
                    {panelData.stress_factors.map((f, i) => (
                      <div key={i} style={{
                        display: 'flex', justifyContent: 'space-between',
                        padding: '5px 12px', borderBottom: '1px solid var(--border)'
                      }}>
                        <span style={{ color: 'var(--text-3)' }}>{f.parameter}</span>
                        <span style={{
                          fontFamily: "'JetBrains Mono',monospace", fontWeight: 700,
                          color: String(f.value).includes('High') || String(f.value).includes('Critical')
                            ? 'var(--crimson)' : 'var(--text-1)'
                        }}>{f.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* ── FORECAST CHART ────────────────────────────── */}
            <div className="area-forecast panel">
              <div className="panel-hdr">
                <h3>&#11015; Demand Forecasting — Hourly (24H)</h3>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.65rem', color: 'var(--text-3)' }}>
                  Bi-LSTM Q&#8325;–Q&#8329;&#8325; vs Actual
                </span>
              </div>
              <div className="forecast-body">
                {activeView === 'demand' ? (
                  <>
                    <div className="chart-legend">
                      <div className="legend-item">
                        <div className="legend-dot" style={{ background: 'var(--emerald)' }} />
                        Actual Load
                      </div>
                      <div className="legend-item">
                        <div className="legend-dot" style={{ background: 'var(--amber)' }} />
                        Predicted (Q&#8329;&#8325;)
                      </div>
                      <div className="legend-item">
                        <div className="legend-dot" style={{ background: 'var(--cyan)', opacity: 0.6 }} />
                        Q&#8325; Band
                      </div>
                    </div>
                    <div className="forecast-chart-wrap">
                      {demandLoading ? (
                        <div className="skeleton-block" style={{ height: '100%', borderRadius: 8 }} />
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={demandData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                            <defs>
                              <linearGradient id="gAct" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00e676" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#00e676" stopOpacity={0} />
                              </linearGradient>
                              <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ffb300" stopOpacity={0.15} />
                                <stop offset="95%" stopColor="#ffb300" stopOpacity={0} />
                              </linearGradient>
                              <linearGradient id="gQ5" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.08} />
                                <stop offset="95%" stopColor="#00e5ff" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="h" tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                              axisLine={false} tickLine={false} />
                            <YAxis tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                              axisLine={false} tickLine={false} unit=" kWh" width={58} />
                            <Tooltip content={<ForecastTooltip />} />
                            <Area type="monotone" dataKey="q95" name="Q95"
                              stroke="#ffb300" strokeWidth={1} strokeDasharray="3 2"
                              fill="url(#gPred)" dot={false} />
                            <Area type="monotone" dataKey="actual" name="Actual"
                              stroke="#00e676" strokeWidth={1.5} fill="url(#gAct)" dot={false} />
                            <Area type="monotone" dataKey="q5" name="Q5"
                              stroke="#00e5ff" strokeWidth={1} strokeDasharray="3 2"
                              fill="url(#gQ5)" dot={false} />
                            <Area type="monotone" dataKey="pred" name="Predicted"
                              stroke="#ffb300" strokeWidth={1.5} fill="none" strokeDasharray="4 2" dot={false} />
                          </AreaChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                    <div className="ai-insight">
                      <div className="ai-insight-lbl">AI Insight · GridSentinel Engine</div>
                      <div className={`ai-insight-text ${!aiInsight ? 'waiting' : ''}`}>
                        {aiInsight || 'Awaiting query — ask the console about a meter ID or transformer ID to generate an intelligence report.'}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <button
                      onClick={() => { setActiveView('demand'); setShapData(null); }}
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left', color: 'var(--emerald)', fontFamily: "'JetBrains Mono',monospace", fontSize: '0.72rem', padding: '8px 16px', flexShrink: 0 }}
                    >
                      &#8592; Back to Grid Overview
                    </button>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <TheftChart data={meterData} />
                    </div>
                    <ShapPanel shapData={shapData} />
                  </div>
                )}
              </div>
            </div>

            {/* ── MAP ───────────────────────────────────────── */}
            <div className="area-map panel">
              <div className="panel-hdr">
                <h3>&#128205; Load Zone — Koramangala</h3>
              </div>
              <div className="panel-body no-pad" style={{ height: '100%' }}>
                <KoraMap />
              </div>
            </div>

            {/* ── ANOMALY TABLE ─────────────────────────────── */}
            <div className="area-table panel">
              <div className="panel-hdr">
                <h3>&#9888; Anomalous Cluster Feed</h3>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.65rem', color: 'var(--crimson)', letterSpacing: '0.06em' }}>
                  {criticalCount} CRITICAL
                </span>
              </div>
              <div className="panel-body no-pad" style={{ overflowX: 'auto' }}>
                {clustersLoading ? (
                  <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="skeleton-block" style={{ height: 32, borderRadius: 4 }} />
                    ))}
                  </div>
                ) : (
                  <table className="anomaly-table">
                    <thead>
                      <tr>
                        <th>Feeder / Meter ID</th>
                        <th>Risk Score</th>
                        <th>Detection Logic</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {anomalyClusters.map((c, i) => (
                        <tr key={i}
                          onClick={() => handleMeterClick(c.feeder)}
                          style={{ cursor: 'pointer' }}
                        >
                          <td><span className="feeder-id">{c.feeder}</span></td>
                          <td>
                            <span className="risk-score" style={{
                              color: c.risk >= 80 ? 'var(--crimson)' : c.risk >= 50 ? 'var(--amber)' : 'var(--emerald)'
                            }}>{Math.round(c.risk)}</span>
                            <span style={{ color: 'var(--text-3)', fontSize: '0.65rem', marginLeft: 4 }}>/100</span>
                          </td>
                          <td><span className="logic-chip">{c.logic}</span></td>
                          <td><span className={`status-pill ${c.status}`}>{STATUS_LABELS[c.status] ?? c.status.toUpperCase()}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

          </div>
        )}

        {activeNav === 'anomaly' && (
          <AlertView apiHeaders={apiHeaders} />
        )}

        {activeNav === 'demand' && (
          <PredView apiHeaders={apiHeaders} />
        )}

        {activeNav === 'audit' && (
          <AuditView />
        )}

      </div>

      {/* ── Chat Drawer ─────────────────────────────────────── */}
      <div className={`chat-drawer ${chatOpen ? 'open' : ''}`}>
        <div className="chat-drawer-hdr">
          <span className="chat-drawer-title">&#9889; Query Console</span>
          <button className="drawer-close-btn" onClick={() => setChatOpen(false)}>&#10005;</button>
        </div>
        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>{m.content}</div>
          ))}
          {loading && <div className="chat-loading">Retrieving ML models...</div>}
          <div ref={messagesEndRef} />
        </div>
        <form className="chat-form" onSubmit={handleSend}>
          <input
            id="chat-input"
            ref={chatInputRef}
            className="chat-input"
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            autoComplete="off"
            placeholder="MTR_000042 anomaly | DT_1 stress..."
            disabled={loading}
          />
          <button className="chat-send" type="submit" disabled={loading || !input.trim()}>TX</button>
        </form>
      </div>

    </div>
  );
}
