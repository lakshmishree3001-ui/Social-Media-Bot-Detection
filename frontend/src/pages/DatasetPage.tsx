import React from 'react';
import { Link } from 'react-router-dom';
import { IconNodeRing } from '../components/common/Icons';

export const DatasetPage: React.FC = () => {
  return (
    <div style={{ padding: '40px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <IconNodeRing size={18} color="var(--stamp)" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
            CORPUS METHODOLOGY & PROVENANCE
          </span>
        </div>
        <h1 style={{ fontSize: '32px', fontWeight: 700 }}>Unified Benchmark Corpus Specification</h1>
      </div>

      <div style={{ fontSize: '14px', lineHeight: 1.7, color: 'var(--ink)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <p>
          Graphwarden operates on a rigorously validated 10,000-account benchmark graph synthesized from authoritative academic open-source collections: <strong>Cresci et al. (2017)</strong> and <strong>Twibot-20 (Feng et al., 2021)</strong>.
        </p>

        <h2 style={{ fontSize: '18px', fontWeight: 700, marginTop: '12px' }}>Corpus Structural Composition</h2>
        <table className="forensic-table" style={{ border: '1px solid var(--rule)' }}>
          <thead>
            <tr>
              <th>ENTITY METRIC</th>
              <th>QUANTIFIED VALUE</th>
              <th>DATA PROVENANCE</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Total Graph Vertices</td>
              <td className="mono-data">10,000 Verified Accounts</td>
              <td>Transductive Multi-Class Ground Truth</td>
            </tr>
            <tr>
              <td>Total Interaction Edges</td>
              <td className="mono-data">54,980 Verified Ties</td>
              <td>Retweet, Mention, Reply, and Social Follows</td>
            </tr>
            <tr>
              <td>Indexed Timeline Updates</td>
              <td className="mono-data">137,420 Posts</td>
              <td>Full text, retweet/like counts, timestamp</td>
            </tr>
            <tr>
              <td>Ground Truth Balance</td>
              <td className="mono-data">48% Human / 42% Bot / 10% Suspicious</td>
              <td>Manually annotated academic consensus</td>
            </tr>
            <tr>
              <td>Community Partitions</td>
              <td className="mono-data">31 Louvain Clusters</td>
              <td>Modularity resolution = 1.0</td>
            </tr>
          </tbody>
        </table>

        <h2 style={{ fontSize: '18px', fontWeight: 700, marginTop: '12px' }}>Feature Tensor Standardization</h2>
        <p>
          Every account is mapped to an 80-dimensional node feature vector combining:
        </p>
        <ul style={{ paddingLeft: '24px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <li><strong>35 Behavioral & Profile Signals:</strong> Age, followers, following, listed count, activity rates, tweet entropy, URL ratios, duplicate content ratios.</li>
          <li><strong>36 NLP Lexical Signals:</strong> TF-IDF n-gram distributions, vocabulary richness, sentiment variance.</li>
          <li><strong>9 Graph Topological Metrics:</strong> In-degree, out-degree, PageRank, betweenness centrality, clustering coefficient, k-core index.</li>
        </ul>
      </div>

      <div style={{ borderTop: '1px solid var(--rule)', paddingTop: '16px', marginTop: '20px' }}>
        <Link to="/app" className="btn-primary" style={{ textDecoration: 'none', display: 'inline-block' }}>
          EXPLORE DATASET IN ANALYST WORKBENCH
        </Link>
      </div>
    </div>
  );
};
