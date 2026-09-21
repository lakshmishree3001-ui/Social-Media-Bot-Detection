import React, { useEffect, useState } from 'react';
import { api, CommunitySummary } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { IconNodeRing, IconStampSeal } from '../components/common/Icons';

// Configurable minimum community size before a cluster can trigger high-priority threat status or bot farm designation
export const MIN_THREAT_COMMUNITY_SIZE = 5;

export const CommunitiesView: React.FC = () => {
  const [communities, setCommunities] = useState<CommunitySummary[]>([]);
  const [selectedCommId, setSelectedCommId] = useState<number | null>(null);
  const [commDetail, setCommDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const { openDossierForAccount, selectedDatasetId } = useAppStore();

  useEffect(() => {
    setLoading(true);
    api.getCommunities('bot_concentration', 'desc', selectedDatasetId)
      .then((data) => {
        setCommunities(data);
        if (data.length > 0) {
          // Select genuine top threat community with size >= MIN_THREAT_COMMUNITY_SIZE
          const eligible = data
            .filter(c => c.size >= MIN_THREAT_COMMUNITY_SIZE && (c.classification?.toLowerCase().includes('bot farm') || c.bot_concentration >= 0.50))
            .sort((a, b) => b.bot_concentration - a.bot_concentration || b.size - a.size);
          const topThreat = eligible[0] || data[0];
          setSelectedCommId(topThreat.community_id);
        } else {
          setSelectedCommId(null);
          setCommDetail(null);
        }
      })
      .catch((err) => console.error('Failed to load communities:', err))
      .finally(() => setLoading(false));
  }, [selectedDatasetId]);

  useEffect(() => {
    if (selectedCommId === null) {
      setCommDetail(null);
      return;
    }
    api.getCommunityDetail(selectedCommId, selectedDatasetId)
      .then(setCommDetail)
      .catch((err) => console.error('Failed to load community detail:', err));
  }, [selectedCommId, selectedDatasetId]);

  // Determine top priority threat community: MUST have size >= MIN_THREAT_COMMUNITY_SIZE
  const eligibleThreats = communities
    .filter(c => c.size >= MIN_THREAT_COMMUNITY_SIZE && (c.classification?.toLowerCase().includes('bot farm') || c.bot_concentration >= 0.50))
    .sort((a, b) => b.bot_concentration - a.bot_concentration || b.size - a.size);

  const topThreatCommunity = eligibleThreats.length > 0 ? eligibleThreats[0] : null;
  const highPriorityCount = communities.filter(c => c.size >= MIN_THREAT_COMMUNITY_SIZE && c.bot_concentration >= 0.50).length;

  const getEffectiveClassification = (c: CommunitySummary) => {
    if (c.size < MIN_THREAT_COMMUNITY_SIZE) {
      return c.bot_concentration >= 0.50 ? 'ISOLATED BOT / MICRO-CLUSTER' : 'ISOLATED MICRO-CLUSTER';
    }
    return c.classification?.toUpperCase() || 'MIXED';
  };

  const getBadgeClass = (c: CommunitySummary) => {
    if (c.size < MIN_THREAT_COMMUNITY_SIZE) {
      return 'badge-suspicious';
    }
    if (c.classification === 'Bot Farm' || c.classification?.includes('Bot Farm')) return 'badge-bot';
    if (c.classification === 'Authentic Cluster' || c.classification?.includes('Human')) return 'badge-human';
    return 'badge-suspicious';
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Page Header */}
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <IconNodeRing size={18} color="var(--stamp)" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
            TOPOLOGICAL CLUSTERING // LOUVAIN ALGORITHM
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 700 }}>
          Community Partitions & Bot Farm Detection
        </h1>
        <p style={{ color: 'var(--ink-muted)', maxWidth: '800px', marginTop: '4px' }}>
          The interaction graph decomposes into {communities.length} Louvain modularity {communities.length === 1 ? 'community' : 'communities'}. High modularity subgraphs with elevated bot density reveal automated bot syndicates and coordinated astroturfing clusters.
        </p>
      </div>

      {/* Flagged Alert Box: Dynamic Top Threat Community (Size >= 5) */}
      {topThreatCommunity ? (
        <div style={{
          backgroundColor: 'var(--stamp-bg)',
          border: '1.5px solid var(--stamp)',
          padding: '16px 20px',
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '24px',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <IconStampSeal size={18} color="var(--stamp)" />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 700, color: 'var(--stamp)' }}>
                HIGH PRIORITY THREAT DETECTED: COMMUNITY #{topThreatCommunity.community_id} ({getEffectiveClassification(topThreatCommunity)})
              </span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--ink)', maxWidth: '850px' }}>
              Community #{topThreatCommunity.community_id} contains {topThreatCommunity.size} interconnected account{topThreatCommunity.size !== 1 ? 's' : ''} exhibiting <strong>{(topThreatCommunity.bot_concentration * 100).toFixed(1)}% automated bot concentration</strong> and internal edge density {topThreatCommunity.internal_density.toFixed(4)}. Graph Attention weights show dense mutual amplification rings targeting coordinated narratives.
            </p>
          </div>

          <button
            onClick={() => setSelectedCommId(topThreatCommunity.community_id)}
            className="btn-stamp"
            style={{ whiteSpace: 'nowrap' }}
          >
            INSPECT COMMUNITY #{topThreatCommunity.community_id}
          </button>
        </div>
      ) : (
        !loading && communities.length > 0 && (
          <div style={{
            backgroundColor: 'var(--paper-deep)',
            border: '1px solid var(--rule)',
            padding: '14px 20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}>
            <IconNodeRing size={18} color="var(--ledger)" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--ink)' }}>
              NO HIGH-SEVERITY BOT FARMS DETECTED: All {communities.length} partition subgraph{communities.length !== 1 ? 's' : ''} exhibit nominal engagement densities.
            </span>
          </div>
        )
      )}

      {/* Main Split: Communities List & Selected Detail */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        
        {/* Left: Communities Modularity Table */}
        <div style={{ border: '1px solid var(--rule)', backgroundColor: 'var(--paper)' }}>
          <div style={{
            padding: '10px 16px',
            backgroundColor: 'var(--paper-deep)',
            borderBottom: '1px solid var(--rule)',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 600,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            <span>{communities.length} DETECTED GRAPH {communities.length === 1 ? 'COMMUNITY' : 'COMMUNITIES'} (LOUVAIN RESOLUTION = 1.0)</span>
            <span style={{ color: highPriorityCount > 0 ? 'var(--stamp)' : 'var(--ledger)' }}>
              {highPriorityCount} HIGH PRIORITY (SIZE ≥ {MIN_THREAT_COMMUNITY_SIZE})
            </span>
          </div>

          <div style={{
            padding: '8px 16px',
            backgroundColor: 'var(--paper)',
            borderBottom: '1px solid var(--rule-light)',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--ink-muted)',
          }}>
            STATUS: <strong style={{ color: highPriorityCount > 0 ? 'var(--stamp)' : 'var(--ledger)' }}>{highPriorityCount} OF {communities.length}</strong> {communities.length === 1 ? 'COMMUNITY' : 'COMMUNITIES'} CLEARED HIGH-PRIORITY THREAT THRESHOLD (SIZE ≥ {MIN_THREAT_COMMUNITY_SIZE}, BOT DENSITY ≥ 50.0%)
          </div>

          <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
            <table className="forensic-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>CLASSIFICATION</th>
                  <th>SIZE</th>
                  <th>BOT CONCENTRATION</th>
                  <th>INTERNAL DENSITY</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '20px', fontFamily: 'var(--font-mono)' }}>
                      CALCULATING MODULARITY PARTITIONS...
                    </td>
                  </tr>
                ) : communities.length === 0 ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '24px', fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
                      NO GRAPH COMMUNITIES DETECTED FOR THIS DATASET.
                    </td>
                  </tr>
                ) : (
                  communities.map((c) => {
                    const isSelected = selectedCommId === c.community_id;
                    const effectiveClass = getEffectiveClassification(c);
                    const badgeClass = getBadgeClass(c);
                    return (
                      <tr
                        key={c.community_id}
                        onClick={() => setSelectedCommId(c.community_id)}
                        style={{
                          backgroundColor: isSelected ? 'var(--paper-hover)' : undefined,
                          borderLeft: isSelected ? '3px solid var(--stamp)' : '3px solid transparent',
                          cursor: 'pointer',
                        }}
                      >
                        <td className="mono-data" style={{ fontWeight: 600 }}>#{c.community_id}</td>
                        <td>
                          <span className={badgeClass} style={{ fontSize: '10px' }}>
                            {effectiveClass}
                          </span>
                        </td>
                        <td className="mono-data">{c.size} {c.size === 1 ? 'node' : 'nodes'}</td>
                        <td className="mono-data">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ minWidth: '45px' }}>{(c.bot_concentration * 100).toFixed(1)}%</span>
                            <div style={{
                              width: '64px',
                              height: '6px',
                              backgroundColor: 'var(--paper-deep)',
                              border: '1px solid var(--rule-light)',
                              overflow: 'hidden',
                            }}>
                              <div style={{
                                width: `${Math.min(100, Math.max(0, c.bot_concentration * 100))}%`,
                                height: '100%',
                                backgroundColor: c.bot_concentration >= 0.5 ? 'var(--stamp)' : 'var(--ledger)',
                              }} />
                            </div>
                          </div>
                        </td>
                        <td className="mono-data">{c.internal_density.toFixed(4)}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>


        {/* Right: Selected Community Deep-Dive */}
        <div style={{ border: '1px solid var(--rule)', backgroundColor: 'var(--paper)', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            padding: '10px 16px',
            backgroundColor: 'var(--paper-deep)',
            borderBottom: '1px solid var(--rule)',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 600,
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>COMMUNITY #{selectedCommId} METRIC BREAKDOWN</span>
            <span style={{ color: 'var(--stamp)' }}>
              {commDetail ? (commDetail.size < MIN_THREAT_COMMUNITY_SIZE ? 'ISOLATED MICRO-CLUSTER' : commDetail.classification?.toUpperCase()) : ''}
            </span>
          </div>

          {commDetail ? (
            <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>
              
              {/* Summary Metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                <div style={{ padding: '10px', backgroundColor: 'var(--paper-deep)', border: '1px solid var(--rule-light)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-muted)' }}>TOTAL NODES</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700 }}>{commDetail.size}</div>
                </div>
                <div style={{ padding: '10px', backgroundColor: 'var(--paper-deep)', border: '1px solid var(--rule-light)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--stamp)' }}>BOT COUNT</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--stamp)' }}>{commDetail.n_bots}</div>
                </div>
                <div style={{ padding: '10px', backgroundColor: 'var(--paper-deep)', border: '1px solid var(--rule-light)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ledger)' }}>HUMAN COUNT</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--ledger)' }}>{commDetail.n_humans}</div>
                </div>
                <div style={{ padding: '10px', backgroundColor: 'var(--paper-deep)', border: '1px solid var(--rule-light)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--signal-text)' }}>SUSPICIOUS</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 700, color: 'var(--signal-text)' }}>{commDetail.n_suspicious}</div>
                </div>
              </div>

              {/* Members Table */}
              <div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  fontWeight: 600,
                  marginBottom: '8px',
                  color: 'var(--ink)',
                }}>
                  INDEXED ACCOUNTS IN CLUSTER ({commDetail.members?.length || 0} SAMPLE)
                </div>

                <div style={{ maxHeight: '380px', overflowY: 'auto', border: '1px solid var(--rule-light)' }}>
                  <table className="forensic-table">
                    <thead>
                      <tr>
                        <th>SCREEN NAME</th>
                        <th>ROLE</th>
                        <th>CLASS</th>
                        <th>BOT PROB</th>
                        <th>ACTION</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(commDetail.members || []).map((m: any) => {
                        const mClass = m.predicted_class || m.ground_truth || 'unknown';
                        return (
                          <tr key={m.account_id}>
                            <td style={{ fontWeight: 600 }}>@{m.screen_name}</td>
                            <td className="mono-data">{m.role.toUpperCase()}</td>
                            <td>
                              <span className={mClass === 'bot' ? 'badge-bot' : mClass === 'human' ? 'badge-human' : 'badge-suspicious'}>
                                {mClass.toUpperCase()}
                              </span>
                            </td>
                            <td className="mono-data">{((m.bot_probability || 0.5) * 100).toFixed(1)}%</td>
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
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          ) : (
            <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
              SELECT A COMMUNITY TO VIEW TOPOLOGICAL DETAILS.
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
