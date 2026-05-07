import React, { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

function riskColor(score) {
  if (score < 50) return 'var(--emerald)';
  if (score <= 75) return 'var(--amber)';
  return 'var(--crimson)';
}

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

const ConfidenceTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#111', border: '1px solid #2a2a2a', borderRadius: 6,
      padding: '8px 12px', fontFamily: "'JetBrains Mono',monospace", fontSize: '0.72rem',
    }}>
      <div style={{ color: '#5a5a5a', marginBottom: 6 }}>{label}:00</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || p.stroke, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value} kWh</strong>
        </div>
      ))}
    </div>
  );
};

const DT_OPTIONS = ['DT_1', 'DT_2', 'DT_3', 'DT_4', 'DT_5'];

export default function PredView({ apiHeaders }) {
  const [selectedDt, setSelectedDt] = useState('DT_1');
  const [demandData, setDemandData]  = useState(null);
  const [gsiData, setGsiData]        = useState(null);
  const [loading, setLoading]        = useState(false);

  useEffect(() => {
    const fetchDemand = async () => {
      setLoading(true);
      try {
        const res  = await fetch(`/api/demand/${selectedDt}`, { headers: apiHeaders });
        const data = await res.json();
        setDemandData(data.demand_series || []);
        setGsiData(data);
      } catch {
        setDemandData([]);
        setGsiData(null);
      } finally {
        setLoading(false);
      }
    };
    fetchDemand();
  }, [selectedDt, apiHeaders]);

  const gsiScore = gsiData?.gsi_score != null ? Math.round(gsiData.gsi_score) : 55;
  const color    = riskColor(gsiScore);

  return (
    <div className="full-page-view" style={{ flexDirection: 'row', gap: 0 }}>
      {/* Left: chart area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="panel-hdr">
          <h3>&#11015; Demand Forecasting — 24H Confidence Band</h3>
          <select
            value={selectedDt}
            onChange={e => setSelectedDt(e.target.value)}
            style={{
              background: 'var(--bg-card-2)', border: '1px solid var(--border)',
              color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
              padding: '4px 8px', borderRadius: 'var(--r-sm)', cursor: 'pointer',
            }}
          >
            {DT_OPTIONS.map(dt => <option key={dt} value={dt}>{dt}</option>)}
          </select>
        </div>

        <div style={{ flex: 1, padding: '12px 8px', minHeight: 0 }}>
          {loading ? (
            <div className="skeleton-block" style={{ height: '100%', borderRadius: 8 }} />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={demandData || []} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="gQ95" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ffb300" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#ffb300" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gQ5" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00e5ff" stopOpacity={0.10} />
                    <stop offset="95%" stopColor="#00e5ff" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gAct" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00e676" stopOpacity={0.22} />
                    <stop offset="95%" stopColor="#00e676" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="h"
                  tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                  axisLine={false} tickLine={false} />
                <YAxis
                  tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                  axisLine={false} tickLine={false} unit=" kWh" width={62} />
                <Tooltip content={<ConfidenceTooltip />} />
                {/* Q95 upper band */}
                <Area type="monotone" dataKey="q95" name="Q95 Upper"
                  stroke="#ffb300" strokeWidth={1} strokeDasharray="3 2"
                  fill="url(#gQ95)" dot={false} />
                {/* Actual */}
                <Area type="monotone" dataKey="actual" name="Actual"
                  stroke="#00e676" strokeWidth={2}
                  fill="url(#gAct)" dot={false} />
                {/* Q5 lower band */}
                <Area type="monotone" dataKey="q5" name="Q5 Lower"
                  stroke="#00e5ff" strokeWidth={1} strokeDasharray="3 2"
                  fill="url(#gQ5)" dot={false} />
                {/* Predicted */}
                <Area type="monotone" dataKey="pred" name="Predicted"
                  stroke="#d500f9" strokeWidth={1.5} strokeDasharray="5 3"
                  fill="none" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 16, padding: '0 16px 12px', flexShrink: 0 }}>
          {[
            { color: '#00e676', label: 'Actual Load' },
            { color: '#d500f9', label: 'Predicted' },
            { color: '#ffb300', label: 'Q95 Upper' },
            { color: '#00e5ff', label: 'Q5 Lower' },
          ].map(item => (
            <div key={item.label} className="legend-item">
              <div className="legend-dot" style={{ background: item.color }} />
              {item.label}
            </div>
          ))}
        </div>

        {/* Stress factors table */}
        {gsiData?.stress_factors?.length > 0 && (
          <div style={{ borderTop: '1px solid var(--border)', padding: '10px 0', flexShrink: 0 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)', letterSpacing: '0.1em', padding: '0 16px 6px' }}>
              STRESS FACTORS
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 0 }}>
              {gsiData.stress_factors.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '5px 16px', borderBottom: '1px solid var(--border)',
                  width: '50%', fontSize: '0.72rem',
                }}>
                  <span style={{ color: 'var(--text-3)' }}>{f.parameter}</span>
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontWeight: 700,
                    color: String(f.value).includes('High') || String(f.value).includes('Critical')
                      ? 'var(--crimson)' : 'var(--text-1)',
                  }}>{f.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right: GSI gauge */}
      <div style={{
        width: 220, flexShrink: 0, borderLeft: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', gap: 12, padding: 16,
      }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)', letterSpacing: '0.1em' }}>
          GRID STRESS INDEX
        </div>
        <GsiGauge score={gsiScore} />
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.72rem', fontWeight: 700,
          color, padding: '4px 12px', borderRadius: 'var(--r-sm)',
          background: color + '18', border: `1px solid ${color}44`,
        }}>
          {gsiScore < 35 ? 'STABLE' : gsiScore < 55 ? 'CAUTION' : gsiScore < 75 ? 'STRESSED' : 'CRITICAL'}
        </div>
        {gsiData?.action && (
          <div style={{ textAlign: 'center', fontSize: '0.72rem', color: 'var(--text-2)', lineHeight: 1.5 }}>
            <span style={{ color: 'var(--text-3)', display: 'block', marginBottom: 4, fontFamily: 'var(--font-mono)', fontSize: '0.6rem', letterSpacing: '0.08em' }}>
              DIRECTIVE
            </span>
            <strong style={{ color: 'var(--crimson)' }}>{gsiData.action}</strong>
          </div>
        )}
      </div>
    </div>
  );
}
