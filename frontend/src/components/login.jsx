import React, { useState } from 'react';
import './Login.css';

const IconShield = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);
const IconUser = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="8" r="4" />
    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
  </svg>
);
const IconLock = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);
const IconActivity = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);

export default function Login({ onLogin }) {
  const [formData, setFormData] = useState({ employeeId: '', passkey: '' });
  const [focused, setFocused] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!formData.employeeId || !formData.passkey) {
      setError('All fields are required.');
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.employeeId,
          password: formData.passkey,
        }),
      });
      if (res.status === 401 || !res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.detail || 'Invalid credentials');
        setSubmitting(false);
        return;
      }
      const data = await res.json();
      localStorage.setItem('gs_token', data.access_token);
      setSubmitting(false);
      onLogin(data.access_token);
    } catch {
      setError('Connection error. Is the backend running?');
      setSubmitting(false);
    }
  };

  return (
    <div className="lg-root">
      <div className="lg-grid" />
      <div className="lg-blob lg-blob-1" />
      <div className="lg-blob lg-blob-2" />

      <div className="lg-card">
        <div className="lg-header">
          <div className="lg-icon-wrap"><IconShield /></div>
          <h1 className="lg-title">GRIDSENTINEL OS</h1>
          <p className="lg-subtitle">Sector Alpha · Secure Gateway</p>
        </div>

        <div className="lg-badge">
          <IconActivity />
          BESCOM RESTRICTED — AUTHORIZED PERSONNEL ONLY
        </div>

        <form onSubmit={handleSubmit}>
          <div className="lg-field">
            <label className="lg-label">Employee ID</label>
            <div className={`lg-input-wrap ${focused === 'id' ? 'focused' : ''}`}>
              <span className="lg-input-icon"><IconUser /></span>
              <input
                className="lg-input"
                type="text"
                placeholder="BE-XXXX"
                autoComplete="username"
                value={formData.employeeId}
                onFocus={() => setFocused('id')}
                onBlur={() => setFocused(null)}
                onChange={e => setFormData(p => ({ ...p, employeeId: e.target.value }))}
              />
            </div>
          </div>

          <div className="lg-field">
            <label className="lg-label">Secure Passkey</label>
            <div className={`lg-input-wrap ${focused === 'pass' ? 'focused' : ''}`}>
              <span className="lg-input-icon"><IconLock /></span>
              <input
                className="lg-input"
                type="password"
                placeholder="••••••••"
                autoComplete="current-password"
                value={formData.passkey}
                onFocus={() => setFocused('pass')}
                onBlur={() => setFocused(null)}
                onChange={e => setFormData(p => ({ ...p, passkey: e.target.value }))}
              />
            </div>
          </div>

          {error && <div className="lg-error">&#9888; {error}</div>}

          <button className="lg-btn" type="submit" disabled={submitting}>
            {submitting ? (
              <span className="lg-btn-inner">
                <span className="lg-spinner" />
                AUTHENTICATING...
              </span>
            ) : 'Authorize Access'}
          </button>
        </form>

        <div className="lg-footer">
          <div className="lg-status">
            <div className="lg-pulse">
              <div className="lg-pulse-core" />
              <div className="lg-pulse-ring" />
            </div>
            Local AI: Active
          </div>
          <div>BESCOM AUDIT SEC-4.2</div>
        </div>
      </div>
    </div>
  );
}
