import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0d0d0d', border: '1px solid #2a2a2a', borderRadius: 6,
      padding: '8px 12px', fontFamily: "'JetBrains Mono',monospace", fontSize: '0.7rem',
    }}>
      <div style={{ color: 'var(--text-3)', marginBottom: 6 }}>Hour {label}:00</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value} kWh</strong>
        </div>
      ))}
    </div>
  );
};

export default function TheftChart({ data }) {
  if (!data || !data.dtw_series) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'var(--text-3)',
        fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
      }}>
        No telemetry — click a meter row to load DTW analysis.
      </div>
    );
  }

  const cassScore  = data.cass_score ?? 0;
  const cassColor  = cassScore >= 80 ? 'var(--crimson)' : cassScore >= 60 ? 'var(--amber)' : cassScore >= 35 ? '#ffd54f' : 'var(--emerald)';
  const cassLabel  = cassScore >= 80 ? 'IMMEDIATE' : cassScore >= 60 ? 'INSPECT' : cassScore >= 35 ? 'WATCH' : 'NORMAL';

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 0 }}>

      {/* Header row */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0,
      }}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            DTW Divergence Analysis
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 700, color: 'var(--cyan)', marginTop: 2 }}>
            {data.meter_id}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-3)', letterSpacing: '0.1em' }}>
            CASS SCORE
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: cassColor, marginTop: 2 }}>
            {cassScore.toFixed(1)}
            <span style={{ fontSize: '0.6rem', color: 'var(--text-3)', marginLeft: 3 }}>/100</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: cassColor, letterSpacing: '0.1em', marginTop: 1 }}>
            {cassLabel}
          </div>
        </div>
      </div>

      {/* Chart — fills remaining space */}
      <div style={{ flex: 1, minHeight: 0, padding: '8px 4px 4px 4px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.dtw_series} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis
              dataKey="hour"
              tick={{ fill: 'var(--text-3)', fontSize: 9, fontFamily: "'JetBrains Mono',monospace" }}
              axisLine={false} tickLine={false}
              tickFormatter={v => `${v}h`}
            />
            <YAxis
              tick={{ fill: 'var(--text-3)', fontSize: 9, fontFamily: "'JetBrains Mono',monospace" }}
              axisLine={false} tickLine={false}
              tickFormatter={v => `${v}`}
              width={36}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.65rem', paddingTop: 4 }}
            />
            <Line
              type="monotone" dataKey="peer_avg" name="Peer Avg"
              stroke="var(--text-3)" strokeDasharray="5 4"
              dot={false} strokeWidth={1.5}
            />
            <Line
              type="monotone" dataKey="target" name="This Meter"
              stroke="var(--crimson)"
              dot={false} strokeWidth={2}
              activeDot={{ r: 4, fill: 'var(--crimson)' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '6px 16px', borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-mono)', fontSize: '0.65rem', flexShrink: 0,
      }}>
        <span style={{ color: 'var(--text-3)' }}>Detection: <span style={{ color: 'var(--emerald)' }}>DTW + CASS ENGINE</span></span>
        <span style={{ color: 'var(--text-3)' }}>
          Divergence: <span style={{ color: cassColor }}>
            {data.dtw_series.length > 0
              ? Math.abs(
                  (data.dtw_series.reduce((s, r) => s + (r.target - r.peer_avg), 0) / data.dtw_series.length)
                ).toFixed(2)
              : '—'} kWh avg
          </span>
        </span>
      </div>
    </div>
  );
}
