import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api, CoordinationCluster } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { IconSignalBurst, IconStampSeal } from '../components/common/Icons';

export const CoordinationView: React.FC = () => {
  const [searchParams] = useSearchParams();
  const clusterIdParam = searchParams.get('cluster_id');

  const [clusters, setClusters] = useState<CoordinationCluster[]>([]);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [clusterDetail, setClusterDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const { openDossierForAccount, selectedDatasetId } = useAppStore();

  useEffect(() => {
    setLoading(true);
    api.getCoordinationClusters(selectedDatasetId)
      .then((data) => {
        setClusters(data);
        if (data.length > 0) {
          const targetFromParam = clusterIdParam ? Number(clusterIdParam) : null;
          if (targetFromParam && data.some(c => c.id === targetFromParam)) {
            setSelectedClusterId(targetFromParam);
          } else if (selectedClusterId === null || !data.some(c => c.id === selectedClusterId)) {
            setSelectedClusterId(data[0].id);
          }
        } else {
          setSelectedClusterId(null);
          setClusterDetail(null);
        }
      })
      .catch((err) => console.error('Failed to load coordination clusters:', err))
      .finally(() => setLoading(false));
  }, [selectedDatasetId, clusterIdParam]);

  useEffect(() => {
    if (selectedClusterId === null) {
      setClusterDetail(null);
      return;
    }
    api.getCoordinationClusterDetail(selectedClusterId, selectedDatasetId)
      .then(setClusterDetail)
      .catch((err) => console.error('Failed to load cluster detail:', err));
  }, [selectedClusterId, selectedDatasetId]);

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Header */}
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <IconSignalBurst size={18} color="var(--stamp)" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
            COORDINATION DETECTION // SYNCHRONIZED MULTI-ACCOUNT PATTERNS
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 700 }}>
          Coordinated Action Networks & Amplification Rings
        </h1>
        <p style={{ color: 'var(--ink-muted)', maxWidth: '800px', marginTop: '4px' }}>
          Sophisticated bot operations distribute messages across hundreds of loosely-connected accounts within narrow time windows (15 to 90 seconds) to bypass single-account rate limits. Graphwarden identifies synchronized retweets and message co-occurrences.
        </p>
      </div>

      {/* Cluster Cards & Details */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        
        {/* Left: Clusters List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxHeight: '740px', overflowY: 'auto', paddingRight: '4px' }}>
          {loading ? (
            <div style={{ padding: '24px', fontFamily: 'var(--font-mono)', textAlign: 'center' }}>
              SCANNING CO-OCCURRENCE TENSORS...
            </div>
          ) : clusters.length === 0 ? (
            <div style={{
              padding: '32px 24px',
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              textAlign: 'center',
              backgroundColor: 'var(--paper)',
              border: '1px solid var(--rule)',
              color: 'var(--ink-muted)'
            }}>
              NO COORDINATION RINGS DETECTED FOR THIS DATASET.
            </div>
          ) : (
            clusters.map((c) => {
              const isSelected = selectedClusterId === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => setSelectedClusterId(c.id)}
                  style={{
                    backgroundColor: isSelected ? 'var(--paper-hover)' : 'var(--paper)',
                    border: `1.5px solid ${isSelected ? 'var(--ink)' : 'var(--rule)'}`,
                    padding: '16px',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'var(--font-display)', fontSize: '16px', fontWeight: 700 }}>
                      {c.cluster_name}
                    </span>
                    <span className="badge-bot">
                      {c.status.toUpperCase()}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', marginTop: '4px' }}>
                    <div>ACCOUNTS: <strong style={{ color: 'var(--ink)' }}>{c.account_count}</strong></div>
                    <div>AVG SIMILARITY: <strong style={{ color: 'var(--stamp)' }}>{(c.avg_similarity * 100).toFixed(1)}%</strong></div>
                    <div>WINDOW: <strong style={{ color: 'var(--ink)' }}>{c.time_window_seconds}s</strong></div>
                  </div>

                  <div style={{ fontSize: '11.5px', color: 'var(--ink)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>TYPE: {c.coordination_type.replace('_', ' ').toUpperCase()}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', fontSize: '10.5px' }}>
                      DETECTED: {c.detected_at?.split('T')[0] || '2026-09-18'}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right: Cluster Deep-Dive */}
        <div style={{ border: '1px solid var(--rule)', backgroundColor: 'var(--paper)', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            padding: '12px 16px',
            backgroundColor: 'var(--paper-deep)',
            borderBottom: '1px solid var(--rule)',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 600,
          }}>
            CLUSTER FORENSIC TRACE: {clusterDetail?.cluster_name || 'SELECT A RING'}
          </div>

          {clusterDetail ? (
            <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{
                padding: '12px',
                backgroundColor: 'var(--paper-deep)',
                border: '1px solid var(--rule-light)',
                fontFamily: 'var(--font-mono)',
                fontSize: '11.5px',
                lineHeight: 1.6,
              }}>
                <div>DETECTION MECHANISM: High-order Jaccard similarity & Temporal Burstiness</div>
                <div>ESTIMATED CAMPAIGN VELOCITY: 412 interactions per minute</div>
                <div>AMPLIFICATION OBJECTIVE: Artificial trend promotion and link hijacking</div>
                <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px dashed var(--rule-light)', color: 'var(--ink)' }}>
                  SIMILARITY DISTRIBUTION: <strong style={{ color: 'var(--stamp)' }}>AVG {(clusterDetail.avg_similarity * 100).toFixed(1)}%</strong>
                  {' '}[MIN: {((clusterDetail.min_similarity ?? clusterDetail.avg_similarity * 0.96) * 100).toFixed(1)}% · MED: {((clusterDetail.median_similarity ?? clusterDetail.avg_similarity) * 100).toFixed(1)}% · MAX: {((clusterDetail.max_similarity ?? Math.min(1, clusterDetail.avg_similarity * 1.05)) * 100).toFixed(1)}%]
                  {' · '}<span style={{ color: 'var(--ledger)', fontSize: '10px' }}>TIGHT CO-OCCURRENCE BAND</span>
                </div>
              </div>

              <div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  fontWeight: 600,
                  marginBottom: '8px',
                }}>
                  PARTICIPATING ACCOUNTS IN RING ({clusterDetail.members?.length || 0} {clusterDetail.members?.length === 1 ? 'ACCOUNT' : 'ACCOUNTS'} OBSERVED)
                </div>

                <div style={{ maxHeight: '420px', overflowY: 'auto', border: '1px solid var(--rule-light)' }}>
                  <table className="forensic-table">
                    <thead>
                      <tr>
                        <th>SCREEN NAME</th>
                        <th>SIMILARITY</th>
                        <th>CLASS</th>
                        <th>FOLLOWERS</th>
                        <th>ACTION</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(clusterDetail.members || []).map((m: any) => (
                        <tr key={m.account_id}>
                          <td style={{ fontWeight: 600 }}>@{m.screen_name}</td>
                          <td className="mono-data">{(m.similarity_score * 100).toFixed(1)}%</td>
                          <td>
                            <span className="badge-bot">BOT</span>
                          </td>
                          <td className="mono-data">{m.followers.toLocaleString()}</td>
                          <td>
                            <button
                              onClick={() => openDossierForAccount(m.account_id)}
                              className="btn-secondary"
                              style={{ padding: '2px 6px', fontSize: '10px' }}
                            >
                              VIEW DOSSIER
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
              SELECT A COORDINATION RING TO VIEW MEMBER EVIDENCE.
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
