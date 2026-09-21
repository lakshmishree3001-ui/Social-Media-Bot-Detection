import React from 'react';
import { Link } from 'react-router-dom';

export const TermsPage: React.FC = () => {
  return (
    <div style={{ padding: '40px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 700 }}>Terms of Forensic Investigation</h1>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', marginTop: '4px' }}>
          LAST AUDITED: SEPTEMBER 2026 // OPEN SOURCE ACADEMIC LICENSE
        </div>
      </div>

      <div style={{ fontSize: '14px', lineHeight: 1.7, color: 'var(--ink)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <p>
          By accessing the Graphwarden platform and its underlying APIs, analysts and researchers agree to conduct all platform interaction investigations strictly for defensive threat intelligence, academic evaluation, and network integrity validation.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700 }}>1. Lawful & Authorized Analysis</h2>
        <p>
          Users may not utilize Graphwarden to harass, doxx, or unlawfully target individuals. The tool is calibrated exclusively to identify coordinated automation and adversarial bot infrastructure.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700 }}>2. Disclaimer of Absolute Attribution</h2>
        <p>
          Outputs are algorithmic estimations provided on an "as-is" basis. Graphwarden disclaims liability for automated administrative decisions taken without verifying the accompanying evidentiary ego graphs and feature attribution tables.
        </p>
      </div>

      <div style={{ borderTop: '1px solid var(--rule)', paddingTop: '16px' }}>
        <Link to="/" className="btn-secondary" style={{ textDecoration: 'none', display: 'inline-block' }}>
          RETURN TO HOME
        </Link>
      </div>
    </div>
  );
};
