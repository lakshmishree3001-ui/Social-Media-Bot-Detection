import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer style={{
      height: '26px',
      backgroundColor: 'var(--paper-deep)',
      borderTop: '1px solid var(--rule)',
      padding: '0 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      fontFamily: 'var(--font-mono)',
      fontSize: '10px',
      color: 'var(--ink-muted)',
      letterSpacing: '0.04em',
      zIndex: 100,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ color: 'var(--ledger)', fontWeight: 700 }}>●</span>
        <span>LAST BACKUP: 2026-09-18 23:45 UTC</span>
        <span>·</span>
        <span>INTEGRITY: <strong style={{ color: 'var(--ink)' }}>SHA-256 VERIFIED</strong></span>
        <span>·</span>
        <span>CO-OCCURRENCE INDEX: <strong style={{ color: 'var(--ink)' }}>OPTIMIZED</strong></span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <span>GRAPH ENGINE: PYTORCH-GEOMETRIC 2.5</span>
        <span>·</span>
        <span>BUILD: <strong style={{ color: 'var(--ink)' }}>v2.4.1-FORENSIC</strong></span>
      </div>
    </footer>
  );
};
