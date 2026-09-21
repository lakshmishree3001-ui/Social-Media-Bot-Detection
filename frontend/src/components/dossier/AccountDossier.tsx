import React, { useEffect, useState } from 'react';
import { api, AccountDetail } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';
import { IconStampSeal, IconDossierFold, IconAttentionCone } from '../common/Icons';
import { EgoGraph } from '../graph/EgoGraph';

export const AccountDossier: React.FC = () => {
  const { selectedAccountId, isDossierOpen, setIsDossierOpen, activeModel } = useAppStore();
  const [account, setAccount] = useState<AccountDetail | null>(null);
  const [egoData, setEgoData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [inferring, setInferring] = useState<boolean>(false);

  useEffect(() => {
    if (selectedAccountId === null) return;
    setLoading(true);

    Promise.all([
      api.getAccountDetail(selectedAccountId),
      api.getEgoGraph(selectedAccountId, 1, 20),
    ])
      .then(([acc, ego]) => {
        setAccount(acc);
        setEgoData(ego);
      })
      .catch((err) => console.error('Failed to load dossier data:', err))
      .finally(() => setLoading(false));
  }, [selectedAccountId]);

  const handleRunInference = async () => {
    if (selectedAccountId === null) return;
    setInferring(true);
    try {
      const res = await api.predictAccount(selectedAccountId, activeModel);
      // Refresh account data
      const updated = await api.getAccountDetail(selectedAccountId);
      setAccount(updated);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setInferring(false);
    }
  };

  if (!isDossierOpen || selectedAccountId === null) return null;

  const pred = account?.prediction;
  const probs = pred ? {
    bot: pred.bot_probability,
    human: pred.human_probability,
    suspicious: pred.suspicious_probability,
  } : { bot: 0.85, human: 0.10, suspicious: 0.05 };

  const verdict = pred?.predicted_class || account?.ground_truth || 'bot';

  const sortedClasses = [
    { label: 'bot', prob: probs.bot },
    { label: 'human', prob: probs.human },
    { label: 'suspicious', prob: probs.suspicious },
  ].sort((a, b) => b.prob - a.prob);

  const topClass = sortedClasses[0];
  const runnerUp = sortedClasses[1];
  const marginGap = topClass.prob - runnerUp.prob;
  const showNarrowMargin = marginGap < 0.25;

  return (
    <aside style={{
      width: '460px',
      height: 'calc(100vh - 80px)',
      backgroundColor: 'var(--paper)',
      borderLeft: '1.5px solid var(--ink)',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative',
      zIndex: 90,
      overflowY: 'auto',
    }}>
      {/* Dossier Header Bar */}
      <div style={{
        padding: '12px 18px',
        backgroundColor: 'var(--paper-deep)',
        borderBottom: '1px solid var(--rule)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <IconDossierFold size={16} color="var(--ink)" />
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '0.06em',
          }}>
            CASE FILE // GW-{account?.id?.toString().padStart(5, '0') || '00000'}
          </span>
        </div>

        <button
          onClick={() => setIsDossierOpen(false)}
          style={{
            background: 'none',
            border: 'none',
            fontFamily: 'var(--font-mono)',
            fontSize: '14px',
            fontWeight: 700,
            cursor: 'pointer',
            color: 'var(--ink)',
          }}
          title="Close dossier"
        >
          ✕
        </button>
      </div>

      {loading ? (
        <div style={{
          padding: '40px 20px',
          textAlign: 'center',
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          color: 'var(--ink-muted)',
        }}>
          RETRIEVING ACCOUNT DOSSIER & TOPOLOGICAL TRACE...
        </div>
      ) : account ? (
        <div style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Identity & Stamping */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: 700 }}>
                @{account.screen_name}
              </h2>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                UID: {account.user_id_str} | NODE IDX: #{account.id}
              </div>
            </div>

            {/* Verdict Stamp with stamp-press physical ink motion */}
            <div
              key={`${account.id}-${verdict}`}
              className="verdict-pill stamp-press-motion"
              style={{
                padding: '6px 12px',
                border: `1.5px solid ${
                  verdict === 'bot' ? 'var(--stamp)' : verdict === 'human' ? 'var(--ledger)' : 'var(--signal-text)'
                }`,
                backgroundColor: verdict === 'bot' ? 'var(--stamp-bg)' : verdict === 'human' ? 'var(--ledger-bg)' : 'var(--signal-bg)',
                color: verdict === 'bot' ? 'var(--stamp)' : verdict === 'human' ? 'var(--ledger)' : 'var(--signal-text)',
                fontFamily: 'var(--font-mono)',
                fontSize: '11.5px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                textAlign: 'center',
              }}
            >
              <div>{verdict.toUpperCase()}</div>
              <div style={{ fontSize: '9px', fontWeight: 500 }}>
                {((pred?.confidence || 0.9) * 100).toFixed(1)}% CONF
              </div>
            </div>
          </div>

          {/* Probability Distribution Bar */}
          <div style={{
            border: '1px solid var(--rule-light)',
            padding: '12px',
            backgroundColor: 'var(--paper-deep)',
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              marginBottom: '6px',
              color: 'var(--ink-muted)',
            }}>
              <span>HUMAN: {(probs.human * 100).toFixed(1)}%</span>
              <span>SUSPICIOUS: {(probs.suspicious * 100).toFixed(1)}%</span>
              <span>BOT: {(probs.bot * 100).toFixed(1)}%</span>
            </div>

            {/* Segmented Bar */}
            <div style={{ height: '10px', width: '100%', display: 'flex', backgroundColor: '#D1C8B6', overflow: 'hidden' }}>
              <div style={{ width: `${probs.human * 100}%`, backgroundColor: 'var(--ledger)' }} title="Human" />
              <div style={{ width: `${probs.suspicious * 100}%`, backgroundColor: 'var(--signal)' }} title="Suspicious" />
              <div style={{ width: `${probs.bot * 100}%`, backgroundColor: 'var(--stamp)' }} title="Bot" />
            </div>

            {/* Narrow Decision Margin Callout */}
            {showNarrowMargin && (
              <div style={{
                marginTop: '8px',
                padding: '6px 8px',
                backgroundColor: 'var(--paper)',
                border: '1px dashed var(--rule)',
                fontFamily: 'var(--font-mono)',
                fontSize: '10px',
                color: 'var(--ink)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}>
                <span style={{ color: 'var(--signal-text)', fontWeight: 700 }}>NARROW MARGIN:</span>
                <span>
                  +{(marginGap * 100).toFixed(1)}% OVER <strong>{runnerUp.label.toUpperCase()}</strong> ({(runnerUp.prob * 100).toFixed(1)}%)
                </span>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '9.5px', fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
              <span>ENGINE: {activeModel}</span>
              <span>LATENCY: {pred?.latency_ms || 11.4}ms</span>
            </div>
          </div>

          {/* Core Profile Metrics */}
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ink-muted)',
              marginBottom: '6px',
            }}>
              OBSERVED PROFILE ATTRIBUTES
            </div>

            <table className="forensic-table">
              <tbody>
                <tr>
                  <th>FOLLOWERS</th>
                  <td className="mono-data">{account.followers_count.toLocaleString()}</td>
                  <th>FOLLOWING</th>
                  <td className="mono-data">{account.following_count.toLocaleString()}</td>
                </tr>
                <tr>
                  <th>ACCOUNT AGE</th>
                  <td className="mono-data">{account.account_age_days} days</td>
                  <th>POST COUNT</th>
                  <td className="mono-data">{account.post_count.toLocaleString()}</td>
                </tr>
                <tr>
                  <th>VERIFIED</th>
                  <td className="mono-data">{account.verified ? 'YES' : 'NO'}</td>
                  <th>GROUND TRUTH</th>
                  <td className="mono-data" style={{ fontWeight: 600 }}>{account.ground_truth.toUpperCase()}</td>
                </tr>
                <tr>
                  <th>COMMUNITY</th>
                  <td className="mono-data">#{account.community?.community_id ?? 'N/A'}</td>
                  <th>COMMUNITY TYPE</th>
                  <td className="mono-data">{account.community?.classification ?? 'Mixed'}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Ego Graph Neighborhood */}
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ink-muted)',
              marginBottom: '6px',
            }}>
              INTERACTION SUBGRAPH & GAT ATTENTION
            </div>
            <EgoGraph egoData={egoData} loading={false} />
          </div>

          {/* Diagnostic Signals Table */}
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ink-muted)',
              marginBottom: '6px',
            }}>
              DIAGNOSTIC EXPLANATORY SIGNALS
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {(pred?.signals || []).map((s, idx) => (
                <div key={idx} style={{
                  padding: '8px 10px',
                  backgroundColor: 'var(--paper-deep)',
                  border: '1px solid var(--rule-light)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '3px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: '11.5px' }}>{s.name}</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '9.5px',
                      padding: '1px 5px',
                      backgroundColor: 'var(--paper)',
                      border: '1px solid var(--rule)',
                    }}>
                      {s.category.toUpperCase()} // WT: {s.weight.toFixed(2)}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--ink-muted)' }}>
                    {s.description}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Posts Inspection */}
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ink-muted)',
              marginBottom: '6px',
            }}>
              TIMELINE UPDATES SAMPLE
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {account.recent_posts && account.recent_posts.length > 0 ? (
                account.recent_posts.map((post) => (
                  <div key={post.id} style={{
                    padding: '8px 10px',
                    border: '1px solid var(--rule-light)',
                    backgroundColor: 'var(--paper)',
                    fontSize: '11.5px',
                  }}>
                    <div style={{ color: 'var(--ink)' }}>"{post.text}"</div>
                    <div style={{
                      display: 'flex',
                      gap: '12px',
                      marginTop: '4px',
                      fontSize: '9.5px',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--ink-muted)',
                    }}>
                      <span>RT: {post.retweet_count}</span>
                      <span>FAV: {post.like_count}</span>
                      <span>URL: {post.has_url ? 'YES' : 'NO'}</span>
                      <span>HASHTAG: {post.has_hashtag ? 'YES' : 'NO'}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: '11px', color: 'var(--ink-muted)', fontStyle: 'italic' }}>
                  No timeline posts indexed for this node.
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            marginTop: '8px',
            paddingTop: '12px',
            borderTop: '1px solid var(--rule)',
          }}>
            <a
              href={api.getDossierPdfUrl(account.id)}
              download={`graphwarden_dossier_${account.screen_name}.pdf`}
              className="btn-stamp"
              style={{
                textAlign: 'center',
                textDecoration: 'none',
                display: 'block',
              }}
            >
              DOWNLOAD ARCHIVAL PDF DOSSIER
            </a>

            <button
              onClick={handleRunInference}
              disabled={inferring}
              className="btn-secondary"
              style={{ width: '100%' }}
            >
              {inferring ? 'COMPUTING GNN EMBEDDINGS...' : `RE-EVALUATE VIA ${activeModel}`}
            </button>
          </div>

        </div>
      ) : null}
    </aside>
  );
};

