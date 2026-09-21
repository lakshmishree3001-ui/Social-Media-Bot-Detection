import React from 'react';
import { Link } from 'react-router-dom';

export const PrivacyPage: React.FC = () => {
  return (
    <div style={{ padding: '40px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 700 }}>Data Privacy & Corpus Anonymization</h1>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', marginTop: '4px' }}>
          COMPLIANT WITH ACADEMIC OPEN DATA GUIDELINES // GDPR ARTICLE 89 EXEMPTION
        </div>
      </div>

      <div style={{ fontSize: '14px', lineHeight: 1.7, color: 'var(--ink)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <p>
          Graphwarden strictly indexes publicly available social network interaction metadata derived from recognized open-access benchmarks (Cresci-2017, Twibot-20).
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700 }}>1. Zero Private Data Ingestion</h2>
        <p>
          Graphwarden does not collect, process, or store private direct messages, IP addresses, geolocation traces, or non-public personal details. All indexed vertices represent publicly observable account nodes.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700 }}>2. Right to Verification & Correction</h2>
        <p>
          If an account holder believes an account has been mischaracterized by automated graph classifiers, human analysts can generate an auditable PDF dossier containing full feature attribution for independent manual adjudication.
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
