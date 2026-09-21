import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { IconSignalBurst } from '../components/common/Icons';

export const TemporalView: React.FC = () => {
  const [temporalData, setTemporalData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const { openDossierForAccount } = useAppStore();
  const navigate = useNavigate();

  useEffect(() => {
    api.getTemporalAnalysis()
      .then(setTemporalData)
      .catch((err) => console.error('Failed to load temporal data:', err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
        COMPUTING DIURNAL POSTING DISTRIBUTIONS & BURST VOLUMES...
      </div>
    );
  }

  const diurnal = temporalData?.diurnal_distribution || [];
  const bursts = temporalData?.burst_events || [];
  const repAccounts = temporalData?.representative_accounts || [];

  const handleInspectAccount = (accId: number) => {
    openDossierForAccount(accId);
    navigate('/app');
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Header */}
      <div style={{ borderBottom: '1.5px solid var(--ink)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <IconSignalBurst size={18} color="var(--signal)" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
            TEMPORAL SIGNAL ANALYSIS // DIURNAL REGULARITY & BURSTINESS
          </span>
        </div>
        <h1 style={{ fontSize: '28px', fontWeight: 700 }}>
          Diurnal Activity Distributions & Synchronized Bursts
        </h1>
        <p style={{ color: 'var(--ink-muted)', maxWidth: '850px', marginTop: '4px' }}>
          Human posting behavior naturally follows a circadian biological rhythm with significant nocturnal lulls (2 AM to 6 AM UTC). Automated bots exhibit unnaturally flat, uniform diurnal curves with high entropy and synchronized micro-burst spikes.
        </p>
      </div>

      {/* Diurnal Profile Comparison Chart */}
      <div style={{
        border: '1px solid var(--rule)',
        backgroundColor: 'var(--paper)',
        padding: '20px',
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 700 }}>
              24-Hour Diurnal Volume Fraction (Normalized UTC)
            </h3>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)' }}>
              Green: Verified Humans | Red: Automated Bots
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            <div>MEAN BOT ENTROPY: <strong style={{ color: 'var(--stamp)' }}>{temporalData?.mean_bot_entropy} (UNIFORM)</strong></div>
            <div>MEAN HUMAN ENTROPY: <strong style={{ color: 'var(--ledger)' }}>{temporalData?.mean_human_entropy} (CIRCADIAN)</strong></div>
          </div>
        </div>

        {/* 24-Hour Bar Chart Visualization */}
        <div style={{
          height: '240px',
          display: 'flex',
          alignItems: 'flex-end',
          gap: '8px',
          paddingBottom: '20px',
          borderBottom: '1px solid var(--rule)',
        }}>
          {diurnal.map((item: any) => {
            const botHeight = (item.bot_fraction / 0.12) * 200;
            const humanHeight = (item.human_fraction / 0.12) * 200;

            return (
              <div key={item.hour} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end' }}>
                <div style={{ display: 'flex', gap: '2px', alignItems: 'flex-end', width: '100%' }}>
                  {/* Human Bar */}
                  <div
                    style={{
                      flex: 1,
                      height: `${Math.max(4, humanHeight)}px`,
                      backgroundColor: 'var(--ledger)',
                      transition: 'height 150ms ease',
                    }}
                    title={`Hour ${item.hour}: Human ${(item.human_fraction * 100).toFixed(1)}%`}
                  />
                  {/* Bot Bar */}
                  <div
                    style={{
                      flex: 1,
                      height: `${Math.max(4, botHeight)}px`,
                      backgroundColor: 'var(--stamp)',
                      transition: 'height 150ms ease',
                    }}
                    title={`Hour ${item.hour}: Bot ${(item.bot_fraction * 100).toFixed(1)}%`}
                  />
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', marginTop: '6px', color: 'var(--ink-muted)' }}>
                  {item.hour.toString().padStart(2, '0')}h
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Small-Multiples 24h Account Diurnal Sparklines */}
      {repAccounts.length > 0 && (
        <div style={{
          border: '1px solid var(--rule)',
          backgroundColor: 'var(--paper)',
        }}>
          <div style={{
            padding: '12px 18px',
            backgroundColor: 'var(--paper-deep)',
            borderBottom: '1px solid var(--rule)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', fontWeight: 600 }}>
                REPRESENTATIVE 24-HOUR ACTIVITY PROFILES // SMALL-MULTIPLES COMPARISON
              </span>
              <span style={{ marginLeft: '12px', fontSize: '11px', color: 'var(--ink-muted)' }}>
                Comparing synthetic round-the-clock polling and burst spikes against authentic circadian cycles.
              </span>
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-muted)' }}>
              2 BOTS · 3 VERIFIED HUMANS
            </span>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '16px',
            padding: '16px',
          }}>
            {repAccounts.map((acc: any) => {
              const isBot = acc.label === 'bot';
              const maxVal = Math.max(...acc.hourly_weights, 1);
              return (
                <div
                  key={acc.id}
                  style={{
                    border: '1px solid var(--rule)',
                    padding: '12px',
                    backgroundColor: 'var(--paper)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '12px', color: 'var(--ink)' }}>
                        {acc.handle}
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: isBot ? 'var(--stamp)' : 'var(--ledger)', fontWeight: 600, marginTop: '2px' }}>
                        {acc.pattern_type}
                      </div>
                    </div>
                    <span className={isBot ? 'badge-bot' : 'badge-human'} style={{ fontSize: '10px', padding: '1px 6px' }}>
                      {acc.label.toUpperCase()}
                    </span>
                  </div>

                  {/* 24-Hour Micro Sparkline */}
                  <div style={{
                    height: '48px',
                    display: 'flex',
                    alignItems: 'flex-end',
                    gap: '1px',
                    backgroundColor: 'var(--paper-deep)',
                    padding: '4px 6px',
                    borderBottom: '1px solid var(--rule)',
                  }}>
                    {acc.hourly_weights.map((val: number, h: number) => {
                      const barH = (val / maxVal) * 38;
                      return (
                        <div
                          key={h}
                          style={{
                            flex: 1,
                            height: `${Math.max(2, barH)}px`,
                            backgroundColor: isBot ? 'var(--stamp)' : 'var(--ledger)',
                            opacity: val > 0 ? 0.9 : 0.2,
                          }}
                          title={`Hour ${h.toString().padStart(2, '0')}:00 UTC — Activity: ${val}`}
                        />
                      );
                    })}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-muted)' }}>
                    <span>00h</span>
                    <span>06h</span>
                    <span>12h</span>
                    <span>18h</span>
                    <span>23h</span>
                  </div>

                  <p style={{ fontSize: '11px', color: 'var(--ink-muted)', lineHeight: '1.4', margin: 0 }}>
                    {acc.description}
                  </p>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto', paddingTop: '6px', borderTop: '1px dashed var(--rule)' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-muted)' }}>
                      ENTROPY: <strong>{acc.entropy}</strong>
                    </span>
                    <button
                      type="button"
                      onClick={() => handleInspectAccount(acc.id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '10px',
                        fontWeight: 600,
                        color: 'var(--ink)',
                        cursor: 'pointer',
                        padding: 0,
                        textDecoration: 'underline',
                      }}
                    >
                      [DOSSIER &rarr;]
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Synchronized Burst Events Table */}
      <div style={{
        border: '1px solid var(--rule)',
        backgroundColor: 'var(--paper)',
      }}>
        <div style={{
          padding: '12px 18px',
          backgroundColor: 'var(--paper-deep)',
          borderBottom: '1px solid var(--rule)',
          fontFamily: 'var(--font-mono)',
          fontSize: '11.5px',
          fontWeight: 600,
        }}>
          DETECTED HIGH-DENSITY TEMPORAL BURSTS (ASTROTURFING & RETWEET STORMS)
        </div>

        <table className="forensic-table">
          <thead>
            <tr>
              <th>EVENT TIMESTAMP (UTC)</th>
              <th>CLASSIFICATION</th>
              <th>TARGET HASHTAG / THEME</th>
              <th>POST VOLUME</th>
              <th>PARTICIPATING ACCOUNTS</th>
              <th>DURATION</th>
              <th>ENTROPY</th>
              <th>COORDINATION ACTION</th>
            </tr>
          </thead>
          <tbody>
            {bursts.map((b: any) => (
              <tr key={b.id}>
                <td className="mono-data">{b.timestamp.replace('T', ' ').replace('Z', '')}</td>
                <td style={{ fontWeight: 600 }}>{b.classification}</td>
                <td className="mono-data" style={{ color: 'var(--stamp)' }}>{b.target_hashtag}</td>
                <td className="mono-data">
                  {b.volume_posts.toLocaleString()} post{b.volume_posts !== 1 ? 's' : ''}
                </td>
                <td className="mono-data">
                  {b.participating_accounts.toLocaleString()} account{b.participating_accounts !== 1 ? 's' : ''}
                </td>
                <td className="mono-data">{b.duration_seconds}s</td>
                <td className="mono-data">{b.entropy_score}</td>
                <td>
                  <Link
                    to={`/app/coordination?cluster_id=${b.cluster_id || 1}`}
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '11px',
                      fontWeight: 600,
                      color: 'var(--ink)',
                      textDecoration: 'underline',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    [INVESTIGATE RING #{b.cluster_id || 1} &rarr;]
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
};
