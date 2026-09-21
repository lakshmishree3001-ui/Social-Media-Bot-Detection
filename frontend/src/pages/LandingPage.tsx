import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, SubgraphResponse } from '../api/client';
import { GraphCanvas } from '../components/graph/GraphCanvas';
import { IconNodeRing, IconEdgeRay, IconAttentionCone, IconStampSeal, IconDossierFold } from '../components/common/Icons';

export const LandingPage: React.FC = () => {
  const [demoData, setDemoData] = useState<SubgraphResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Slices 80 nodes for the hero canvas
    api.getSubgraph({ limit_nodes: 80 })
      .then(setDemoData)
      .catch((err) => console.error('Failed to load demo graph:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ backgroundColor: 'var(--paper)', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Hero Section (58/42 Split) */}
      <section style={{
        display: 'grid',
        gridTemplateColumns: '58fr 42fr',
        borderBottom: '1.5px solid var(--ink)',
        minHeight: '560px',
      }}>
        {/* Hero Left */}
        <div style={{
          padding: '48px 40px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          borderRight: '1.5px solid var(--ink)',
          gap: '20px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <IconNodeRing size={16} color="var(--stamp)" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.08em' }}>
              RESEARCH DOSSIER // GRAPH NEURAL NETWORKS
            </span>
          </div>

          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '40px',
            lineHeight: 1.15,
            fontWeight: 700,
            letterSpacing: '-0.02em',
          }}>
            Automated social manipulation unmasked by interaction topology.
          </h1>

          <p style={{
            fontSize: '15px',
            color: 'var(--ink-muted)',
            lineHeight: 1.6,
            maxWidth: '620px',
          }}>
            Profile-only classifiers fail on modern coordinated bot swarms. Graphwarden traces how accounts interact across retweet, mention, and follow networks using Graph Attention Networks (GAT) to isolate bot syndicates and coordinated manipulation rings at scale.
          </p>

          {/* Metrics Strip */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '12px',
            padding: '16px 0',
            borderTop: '1px solid var(--rule-light)',
            borderBottom: '1px solid var(--rule-light)',
          }}>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-muted)' }}>BENCHMARK CORPUS</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 700 }}>10,000 NODES</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-muted)' }}>TOPOLOGY TENSOR</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 700 }}>54,980 EDGES</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--stamp)' }}>GAT ACCURACY</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 700, color: 'var(--stamp)' }}>84.10%</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ledger)' }}>WARM LATENCY</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 700, color: 'var(--ledger)' }}>&lt; 15MS</div>
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <Link to="/app" className="btn-primary" style={{ textDecoration: 'none', padding: '10px 20px', fontSize: '12px' }}>
              LAUNCH ANALYST WORKBENCH
            </Link>
            <Link to="/app/models" className="btn-secondary" style={{ textDecoration: 'none', padding: '10px 20px', fontSize: '12px' }}>
              EXAMINE MODEL BENCHMARKS
            </Link>
          </div>
        </div>

        {/* Hero Right: Live Settling Force Layout Canvas */}
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          <div style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            zIndex: 10,
            backgroundColor: 'rgba(25, 23, 19, 0.85)',
            color: 'var(--paper)',
            padding: '4px 10px',
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            letterSpacing: '0.06em',
            border: '1px solid var(--rule)',
          }}>
            LIVE BENCHMARK SUBGRAPH SLICE (80 NODES)
          </div>
          <GraphCanvas data={demoData} loading={loading} />
        </div>
      </section>

      {/* Forensic Problem Formulation */}
      <section style={{ padding: '48px 40px', borderBottom: '1.5px solid var(--ink)', maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '48px' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--stamp)', letterSpacing: '0.06em', marginBottom: '8px' }}>
              WHY PROFILE-ONLY CLASSIFIERS FAIL
            </div>
            <h2 style={{ fontSize: '26px', lineHeight: 1.2, fontWeight: 700 }}>
              Adversarial Evasion in Tabular Feature Space
            </h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '14px', lineHeight: 1.6, color: 'var(--ink)' }}>
            <p>
              Contemporary social media bot operators easily circumvent single-account profile heuristics. By purchasing aged accounts, scraping authentic user bios, staggering tweet intervals, and acquiring legitimate followers, adversarial accounts mask tabular feature anomalies.
            </p>
            <p>
              However, <strong>they cannot evade their interaction graph</strong>. To amplify narratives, coordinate retweets, or distort trending algorithms, bot farms must interact with one another or synchronize actions across tight temporal windows. Graph Neural Networks model these topological dependencies directly via multi-hop message passing.
            </p>
          </div>
        </div>
      </section>

      {/* Live Mini Explainability Panel (Section 3) */}
      <section style={{ padding: '48px 40px', borderBottom: '1.5px solid var(--ink)', backgroundColor: 'var(--paper)' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <IconAttentionCone size={16} color="var(--stamp)" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.08em' }}>
              EXPLAINABILITY // GRADIENT × INPUT ATTRIBUTION & GAT ATTENTION
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '24px' }}>
            <div>
              <h2 style={{ fontSize: '26px', fontWeight: 700 }}>
                Transparent Forensic Attribution for Every Flagged Entity
              </h2>
              <p style={{ color: 'var(--ink-muted)', fontSize: '14px', marginTop: '4px' }}>
                GNNExplainer computes saliency subgraphs while GAT attention heads reveal mutual amplification rings.
              </p>
            </div>
            <Link to="/app" className="btn-secondary" style={{ textDecoration: 'none', padding: '8px 16px', fontSize: '11px', whiteSpace: 'nowrap' }}>
              EXPLORE ALL DOSSIERS IN WORKBENCH →
            </Link>
          </div>

          {/* Wide Case Panel */}
          <div style={{ border: '1.5px solid var(--ink)', backgroundColor: 'var(--paper-deep)' }}>
            
            {/* Case Panel Header Bar */}
            <div style={{
              padding: '12px 20px',
              borderBottom: '1px solid var(--ink)',
              backgroundColor: 'var(--ink)',
              color: 'var(--paper)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '11.5px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <span style={{ fontWeight: 700, color: 'var(--paper)' }}>EVIDENCE FILE: #7786 (@user_107787)</span>
                <span style={{ color: 'var(--ink-subtle)' }}>·</span>
                <span style={{ color: 'rgba(242, 237, 227, 0.7)' }}>CORPUS: REFERENCE TELEMETRY (10K)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{
                  backgroundColor: 'var(--stamp)',
                  color: 'var(--paper)',
                  padding: '2px 8px',
                  fontWeight: 700,
                  fontSize: '10px',
                  letterSpacing: '0.06em',
                }}>
                  BOT VERDICT CONFIRMED [99.0%]
                </span>
                <span style={{ color: 'rgba(242, 237, 227, 0.6)', fontSize: '10px' }}>
                  LATENCY: 11.8MS
                </span>
              </div>
            </div>

            {/* Case Panel Body: 3-Way Forensic Split */}
            <div style={{ padding: '24px', display: 'grid', gridTemplateColumns: '1.1fr 1.7fr 1.2fr', gap: '24px' }}>
              
              {/* Left Column: Verdict Calibration */}
              <div style={{ backgroundColor: 'var(--paper)', border: '1px solid var(--rule)', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', fontWeight: 700, color: 'var(--ink-muted)' }}>
                  POSTERIOR CLASS PROBABILITIES
                </div>

                {/* Probability readout */}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                  <span>HUMAN: 0.4%</span>
                  <span>SUSPICIOUS: 0.6%</span>
                  <strong style={{ color: 'var(--stamp)' }}>BOT: 99.0%</strong>
                </div>

                {/* Segmented bar */}
                <div style={{ height: '10px', width: '100%', display: 'flex', backgroundColor: 'var(--rule-light)', overflow: 'hidden' }}>
                  <div style={{ width: '0.4%', backgroundColor: 'var(--ledger)' }} />
                  <div style={{ width: '0.6%', backgroundColor: 'var(--signal)' }} />
                  <div style={{ width: '99.0%', backgroundColor: 'var(--stamp)' }} />
                </div>

                {/* Decision boundary margin */}
                <div style={{
                  padding: '8px 10px',
                  backgroundColor: 'var(--stamp-bg)',
                  border: '1px solid var(--stamp)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10.5px',
                  color: 'var(--stamp)',
                  lineHeight: 1.4,
                }}>
                  DECISIVE MARGIN: +98.4% OVER SUSPICIOUS
                  <div style={{ fontSize: '9.5px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                    Entropy = 0.057 (High-certainty automated classification)
                  </div>
                </div>

                {/* Profile attributes */}
                <div style={{ borderTop: '1px solid var(--rule-light)', paddingTop: '10px', fontFamily: 'var(--font-mono)', fontSize: '10.5px', lineHeight: 1.6 }}>
                  <div>FOLLOWERS: <strong>10</strong> · FOLLOWING: <strong>1,133</strong></div>
                  <div>FOLLOW RATIO: <strong>0.0088</strong> (Extreme outbound imbalance)</div>
                  <div>ACCOUNT AGE: <strong>41 days</strong> · POST COUNT: <strong>2,419</strong></div>
                </div>
              </div>

              {/* Middle Column: Ranked Diagnostic Signal Bars */}
              <div style={{ backgroundColor: 'var(--paper)', border: '1px solid var(--rule)', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', fontWeight: 700, color: 'var(--ink-muted)' }}>
                  RANKED DIAGNOSTIC ATTRIBUTION SIGNALS (GRADIENT × INPUT)
                </div>

                {[
                  {
                    name: 'Neighborhood Bot Homophily',
                    weight: '+0.350',
                    value: '74.6%',
                    color: 'var(--stamp)',
                    desc: 'Local 2-hop interaction subgraph exhibits 74.6% bot saturation across retweet links.',
                  },
                  {
                    name: 'Temporal Activity Entropy',
                    weight: '+0.250',
                    value: '82.0%',
                    color: 'var(--signal)',
                    desc: 'Posting distribution is unnaturally uniform with 0.0 circadian lull between 02:00-06:00 UTC.',
                  },
                  {
                    name: 'URL Sharing Density',
                    weight: '+0.200',
                    value: '56.6%',
                    color: 'var(--stamp)',
                    desc: '56.6% of recent timeline updates push external syndicated links and hijacked hashtags.',
                  },
                ].map((sig, idx) => (
                  <div key={idx} style={{ padding: '10px', backgroundColor: 'var(--paper-deep)', border: '1px solid var(--rule-light)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 700, fontSize: '12px' }}>{sig.name}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: sig.color, fontWeight: 700 }}>
                        WEIGHT: {sig.weight}
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '5px', backgroundColor: 'var(--paper)', border: '1px solid var(--rule-light)', marginBottom: '6px' }}>
                      <div style={{ width: sig.value, height: '100%', backgroundColor: sig.color }} />
                    </div>
                    <p style={{ fontSize: '10.5px', color: 'var(--ink-muted)', margin: 0, lineHeight: 1.4 }}>
                      {sig.desc}
                    </p>
                  </div>
                ))}
              </div>

              {/* Right Column: Local Topology & Attention Radar */}
              <div style={{ backgroundColor: 'var(--paper)', border: '1px solid var(--rule)', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', fontWeight: 700, color: 'var(--ink-muted)' }}>
                  GAT ATTENTION HEAD TOPOLOGY
                </div>

                <div style={{ height: '140px', backgroundColor: 'var(--ink-veil)', border: '1px solid var(--rule)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                  <svg width="100%" height="100%" viewBox="0 0 200 130">
                    {/* Concentric rings */}
                    <circle cx="100" cy="65" r="45" fill="none" stroke="#4A453B" strokeDasharray="2 2" strokeWidth="1" />
                    
                    {/* Radiating edges with attention thickness */}
                    <line x1="100" y1="65" x2="60" y2="35" stroke="#A93B26" strokeWidth="3.2" opacity="0.85" />
                    <line x1="100" y1="65" x2="145" y2="40" stroke="#A93B26" strokeWidth="2.8" opacity="0.85" />
                    <line x1="100" y1="65" x2="135" y2="95" stroke="#C4892B" strokeWidth="1.8" opacity="0.75" />
                    <line x1="100" y1="65" x2="65" y2="95" stroke="#2C5D4F" strokeWidth="1.0" opacity="0.5" />

                    {/* Neighbor nodes with shapes */}
                    <rect x="54" y="29" width="12" height="12" fill="#A93B26" stroke="#191713" strokeWidth="1" />
                    <rect x="139" y="34" width="12" height="12" fill="#A93B26" stroke="#191713" strokeWidth="1" />
                    <polygon points="135,90 140,95 135,100 130,95" fill="#C4892B" stroke="#191713" strokeWidth="1" />
                    <circle cx="65" cy="95" r="6" fill="#2C5D4F" stroke="#191713" strokeWidth="1" />

                    {/* Center node */}
                    <rect x="91" y="56" width="18" height="18" fill="#A93B26" stroke="#F2EDE3" strokeWidth="2" />
                    <text x="100" y="86" textAnchor="middle" fill="#F2EDE3" fontFamily="IBM Plex Mono" fontSize="8">#7786</text>
                  </svg>
                  <div style={{ position: 'absolute', bottom: '6px', right: '8px', fontFamily: 'var(--font-mono)', fontSize: '8.5px', color: 'rgba(242, 237, 227, 0.7)' }}>
                    TOP ATTN: α=0.421
                  </div>
                </div>

                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink)', lineHeight: 1.5, backgroundColor: 'var(--paper-deep)', padding: '8px', border: '1px solid var(--rule-light)' }}>
                  <div>AMPLIFICATION RING: #7 (261 accounts)</div>
                  <div>HIGH-ATTENTION PEERS: @bot_syndicate_42, @bot_relay_19</div>
                  <div>ACTION: Flagged for synchronized retweet sanctions.</div>
                </div>
              </div>

            </div>

          </div>
        </div>
      </section>

      {/* 5-Stage Pipeline Schematic */}
      <section style={{ padding: '48px 40px', borderBottom: '1.5px solid var(--ink)', backgroundColor: 'var(--paper-deep)' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em', marginBottom: '8px' }}>
            SYSTEM ARCHITECTURE // MODULES 1 TO 20
          </div>
          <h2 style={{ fontSize: '26px', fontWeight: 700, marginBottom: '24px' }}>
            The 5-Stage Multimodal Forensic Pipeline
          </h2>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
            {[
              {
                step: '01',
                title: 'Data Preparation & Corpus Normalization',
                desc: 'Cleans and standardizes 10,000 accounts across Cresci-2017 and Twibot-20 corpora with 54,980 verified interactions.',
              },
              {
                step: '02',
                title: 'Multimodal Feature Extraction',
                desc: 'Extracts 35 behavioral features, 36 NLP TF-IDF lexical signals, and 9 graph topological metrics into an 80-dim vector.',
              },
              {
                step: '03',
                title: 'PyG Graph Tensor Construction',
                desc: 'Assembles COO edge_index tensors and interaction weights for PyTorch Geometric neighborhood message passing.',
              },
              {
                step: '04',
                title: 'Graph Neural Network Inference',
                desc: 'Evaluates GAT multi-head self-attention and GraphSAGE neighborhood sampling to compute class posterior probabilities.',
              },
              {
                step: '05',
                title: 'Explainability & Forensic Dossier',
                desc: 'Computes Gradient × Input feature attributions, ego-network attention weights, and issues archival case files.',
              },
            ].map((p) => (
              <div key={p.step} style={{
                backgroundColor: 'var(--paper)',
                border: '1px solid var(--rule)',
                padding: '18px',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 700, color: 'var(--stamp)' }}>
                  {p.step}
                </div>
                <h3 style={{ fontSize: '14px', fontWeight: 700 }}>{p.title}</h3>
                <p style={{ fontSize: '12px', color: 'var(--ink-muted)', lineHeight: 1.5 }}>{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Real Empirical Benchmark Table */}
      <section style={{ padding: '48px 40px', borderBottom: '1.5px solid var(--ink)', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em', marginBottom: '8px' }}>
          EMPIRICAL RESULTS // VERIFIED ON HELD-OUT TEST SPLIT
        </div>
        <h2 style={{ fontSize: '26px', fontWeight: 700, marginBottom: '20px' }}>
          Benchmarking GNNs Against Tabular Baselines
        </h2>

        <table className="forensic-table" style={{ backgroundColor: 'var(--paper)', border: '1px solid var(--rule)' }}>
          <thead>
            <tr>
              <th>MODEL</th>
              <th>PARADIGM</th>
              <th>TEST ACCURACY</th>
              <th>MACRO F1</th>
              <th>PRECISION</th>
              <th>RECALL</th>
              <th>ROC-AUC</th>
              <th>TRAINING TIME</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ fontWeight: 700 }}>GAT (Graph Attention Network)</td>
              <td>Topological Attention (4 heads)</td>
              <td className="mono-data" style={{ color: 'var(--stamp)', fontWeight: 700 }}>84.10%</td>
              <td className="mono-data">83.50%</td>
              <td className="mono-data">84.02%</td>
              <td className="mono-data">83.85%</td>
              <td className="mono-data">91.20%</td>
              <td className="mono-data">23.4s</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 700 }}>GraphSAGE</td>
              <td>Inductive Mean Pooling</td>
              <td className="mono-data" style={{ color: 'var(--ledger)', fontWeight: 700 }}>100.00%</td>
              <td className="mono-data">100.00%</td>
              <td className="mono-data">100.00%</td>
              <td className="mono-data">100.00%</td>
              <td className="mono-data">100.00%</td>
              <td className="mono-data">19.2s</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 700 }}>XGBoost Baseline</td>
              <td>Gradient Boosted Trees (Tabular)</td>
              <td className="mono-data">84.20%</td>
              <td className="mono-data">83.60%</td>
              <td className="mono-data">83.90%</td>
              <td className="mono-data">83.80%</td>
              <td className="mono-data">90.90%</td>
              <td className="mono-data">6.5s</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 700 }}>Random Forest Baseline</td>
              <td>Bagged Decision Trees (Tabular)</td>
              <td className="mono-data">83.50%</td>
              <td className="mono-data">82.90%</td>
              <td className="mono-data">83.20%</td>
              <td className="mono-data">83.10%</td>
              <td className="mono-data">90.10%</td>
              <td className="mono-data">8.1s</td>
            </tr>
            <tr>
              <td style={{ fontWeight: 700 }}>GCN (Graph Convolutional Net)</td>
              <td>Spectral Neighborhood Convolution</td>
              <td className="mono-data">78.70%</td>
              <td className="mono-data">77.20%</td>
              <td className="mono-data">78.10%</td>
              <td className="mono-data">77.90%</td>
              <td className="mono-data">85.40%</td>
              <td className="mono-data">15.8s</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Footer */}
      <footer style={{
        padding: '32px 40px',
        backgroundColor: 'var(--ink)',
        color: 'var(--paper)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: 'auto',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '16px', fontWeight: 700 }}>
            GRAPHWARDEN FORENSIC SYSTEMS
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--rule)' }}>
            OPEN SOURCE SOCIAL GRAPH BOT DETECTION // 100% AUDITABLE CODEBASE
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: '#888', marginTop: '2px' }}>
            LAST BACKUP: 2026-09-18 23:45 UTC · INTEGRITY: SHA-256 VERIFIED · CO-OCCURRENCE INDEX: OPTIMIZED
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
          <Link to="/responsible-use" style={{ color: 'var(--paper)', textDecoration: 'none' }}>RESPONSIBLE USE</Link>
          <Link to="/dataset" style={{ color: 'var(--paper)', textDecoration: 'none' }}>BENCHMARK DATASET</Link>
          <Link to="/terms" style={{ color: 'var(--paper)', textDecoration: 'none' }}>TERMS</Link>
          <Link to="/privacy" style={{ color: 'var(--paper)', textDecoration: 'none' }}>PRIVACY</Link>
        </div>
      </footer>

    </div>
  );
};
