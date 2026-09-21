import React from 'react';
import { Link } from 'react-router-dom';
import { IconStampSeal } from '../components/common/Icons';

export const ResponsibleUsePage: React.FC = () => {
  return (
    <div style={{ padding: '40px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <IconStampSeal size={18} color="var(--stamp)" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
            OPERATIONAL COMPLIANCE & ETHICAL GOVERNANCE
          </span>
        </div>
        <h1 style={{ fontSize: '32px', fontWeight: 700 }}>Responsible Use & Human-in-the-Loop Protocol</h1>
      </div>

      <div style={{ fontSize: '14px', lineHeight: 1.7, color: 'var(--ink)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <p>
          Graphwarden is an investigative instrument designed for cyber threat intelligence, forensic verification, and platform integrity research. It must never be deployed as an autonomous, un-audited enforcement engine for account suspension or censorship without qualified human oversight.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700, marginTop: '12px' }}>1. Probabilistic Uncertainty & Calibration</h2>
        <p>
          All classifications produced by Graph Neural Networks (GAT, GraphSAGE, GCN) are calibrated posterior probabilities, not infallible absolute verdicts. Satirical accounts, aggregated news feeds, automated emergency alerts, and highly active power users often exhibit behavioral entropy patterns resembling automated bot accounts. Every automated classification requires corroborating evidence from the interaction subgraph.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700, marginTop: '12px' }}>2. Mandatory Human Review Safeguards</h2>
        <p>
          No enforcement actions (such as account termination, shadowbanning, or public exposure) should be taken based solely on an automated score. Human analysts must verify:
        </p>
        <ul style={{ paddingLeft: '24px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <li>The presence of coordinated co-occurrence ties across the Louvain community.</li>
          <li>The absence of legitimate API automation (e.g. verified news organizations or public transit bots).</li>
          <li>Attribution weight directions to ensure predictions are not biased by benign follower asymmetry.</li>
        </ul>

        <h2 style={{ fontSize: '18px', fontWeight: 700, marginTop: '12px' }}>3. Archival Auditability</h2>
        <p>
          Every inference request, model activation, and case file export generates a permanent entry in the system audit log. Predictions are immutably tied to the exact MLflow model checkpoint and timestamp to enable retrospective review during appeals.
        </p>
      </div>

      <div style={{ borderTop: '1px solid var(--rule)', paddingTop: '16px', marginTop: '20px' }}>
        <Link to="/app" className="btn-primary" style={{ textDecoration: 'none', display: 'inline-block' }}>
          RETURN TO ANALYST WORKBENCH
        </Link>
      </div>
    </div>
  );
};
