import React, { useState, useEffect } from 'react';
import { api, SubgraphResponse, AccountSummary } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { GraphCanvas } from '../components/graph/GraphCanvas';
import { AccountDossier } from '../components/dossier/AccountDossier';
import { IconSearchReticle, IconNodeRing, IconStampSeal } from '../components/common/Icons';

export const AnalystConsole: React.FC = () => {
  const {
    selectedAccountId,
    openDossierForAccount,
    activeModel,
    setActiveModel,
    isDossierOpen,
    setIsDossierOpen,
    selectedDatasetId,
  } = useAppStore();

  // Filter States
  const [search, setSearch] = useState('');
  const [groundTruth, setGroundTruth] = useState('');
  const [predictedClass, setPredictedClass] = useState('');
  const [graphLimit, setGraphLimit] = useState(250);
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('bot_probability');
  const [sortOrder, setSortOrder] = useState('desc');

  // Data States
  const [subgraphData, setSubgraphData] = useState<SubgraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [accountsData, setAccountsData] = useState<{ total: number; items: AccountSummary[] }>({ total: 0, items: [] });
  const [tableLoading, setTableLoading] = useState(true);

  // Load Subgraph
  useEffect(() => {
    setGraphLoading(true);
    api.getSubgraph({ limit_nodes: graphLimit, ground_truth: groundTruth || undefined }, selectedDatasetId)
      .then(setSubgraphData)
      .catch((err) => console.error('Subgraph load failed:', err))
      .finally(() => setGraphLoading(false));
  }, [graphLimit, groundTruth, selectedDatasetId]);

  // Load Accounts Table
  useEffect(() => {
    setTableLoading(true);
    api.getAccounts({
      page,
      page_size: 20,
      search: search || undefined,
      ground_truth: groundTruth || undefined,
      predicted_class: predictedClass || undefined,
      sort_by: sortBy,
      sort_order: sortOrder,
    }, selectedDatasetId)
      .then((res) => {
        setAccountsData({ total: res.total, items: res.items });
      })
      .catch((err) => console.error('Accounts table load failed:', err))
      .finally(() => setTableLoading(false));
  }, [page, search, groundTruth, predictedClass, sortBy, sortOrder, selectedDatasetId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)', overflow: 'hidden' }}>
      
      {/* Sticky Command Bar (Only shadow in the app) */}
      <div style={{
        backgroundColor: 'var(--paper)',
        borderBottom: '1px solid var(--ink)',
        boxShadow: 'var(--shadow-command-bar)',
        padding: '10px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '16px',
        zIndex: 40,
      }}>
        {/* Left: Search Reticle Input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: '1', maxWidth: '380px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            backgroundColor: 'var(--paper-deep)',
            border: '1px solid var(--rule)',
            padding: '4px 10px',
            width: '100%',
          }}>
            <IconSearchReticle size={14} color="var(--ink-muted)" />
            <input
              type="text"
              placeholder="FILTER ACCOUNT BY SCREEN NAME OR USER ID..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              style={{
                border: 'none',
                background: 'transparent',
                outline: 'none',
                marginLeft: '8px',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                width: '100%',
                color: 'var(--ink)',
              }}
            />
          </div>
        </div>

        {/* Center: Filters */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Classification Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            <span style={{ color: 'var(--ink-muted)' }}>CLASS:</span>
            <select
              value={predictedClass}
              onChange={(e) => {
                setPredictedClass(e.target.value);
                setPage(1);
              }}
              style={{
                padding: '4px 8px',
                backgroundColor: 'var(--paper-deep)',
                border: '1px solid var(--rule)',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                color: 'var(--ink)',
              }}
            >
              <option value="">ALL CLASSIFICATIONS</option>
              <option value="bot">BOT (AUTOMATED)</option>
              <option value="human">HUMAN (AUTHENTIC)</option>
              <option value="suspicious">SUSPICIOUS (COORDINATED)</option>
            </select>
          </div>

          {/* Canvas Nodes Limit */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            <span style={{ color: 'var(--ink-muted)' }}>GRAPH SLICE:</span>
            <select
              value={graphLimit}
              onChange={(e) => setGraphLimit(Number(e.target.value))}
              style={{
                padding: '4px 8px',
                backgroundColor: 'var(--paper-deep)',
                border: '1px solid var(--rule)',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                color: 'var(--ink)',
              }}
            >
              <option value={100}>100 NODES</option>
              <option value={250}>250 NODES</option>
              <option value={500}>500 NODES</option>
            </select>
          </div>

          {/* Model Engine Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            <span style={{ color: 'var(--ink-muted)' }}>MODEL:</span>
            <select
              value={activeModel}
              onChange={(e) => setActiveModel(e.target.value)}
              style={{
                padding: '4px 8px',
                backgroundColor: 'var(--paper-deep)',
                border: '1px solid var(--rule)',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                color: 'var(--ink)',
                fontWeight: 600,
              }}
            >
              <option value="GAT">GAT (ATTENTION)</option>
              <option value="GraphSAGE">GraphSAGE</option>
              <option value="GCN">GCN (CONVOLUTION)</option>
              <option value="XGBoost">XGBoost BASELINE</option>
              <option value="Random Forest">RANDOM FOREST</option>
            </select>
          </div>
        </div>

        {/* Right: Quick Dossier Toggle */}
        <div>
          <button
            onClick={() => setIsDossierOpen(!isDossierOpen)}
            className="btn-secondary"
            style={{ fontSize: '11px', padding: '5px 10px' }}
          >
            {isDossierOpen ? 'COLLAPSE DOSSIER' : 'EXPAND DOSSIER'}
          </button>
        </div>
      </div>

      {/* Main Workspace Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* Left & Middle: Split between Graph Canvas and Forensic Table */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          
          {/* Top Half: Graph Canvas */}
          <div style={{ height: '52%', borderBottom: '1.5px solid var(--ink)', position: 'relative' }}>
            <GraphCanvas
              data={subgraphData}
              loading={graphLoading}
              onSelectNode={(id) => openDossierForAccount(id)}
            />
          </div>

          {/* Bottom Half: Forensic Accounts Table */}
          <div style={{ height: '48%', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--paper)', overflow: 'hidden' }}>
            
            {/* Table Header Controls */}
            <div style={{
              padding: '6px 16px',
              backgroundColor: 'var(--paper-deep)',
              borderBottom: '1px solid var(--rule)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
            }}>
              <div>
                SHOWING <strong>{accountsData.items.length}</strong> OF <strong>{accountsData.total.toLocaleString()}</strong> VERIFIED ACCOUNTS
              </div>

              {/* Sorting Controls */}
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <span>SORT BY:</span>
                <button
                  onClick={() => {
                    setSortBy('bot_probability');
                    setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    fontWeight: sortBy === 'bot_probability' ? 700 : 400,
                    cursor: 'pointer',
                    color: 'var(--ink)',
                  }}
                >
                  BOT PROBABILITY {sortBy === 'bot_probability' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                </button>
                <button
                  onClick={() => {
                    setSortBy('followers_count');
                    setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    fontWeight: sortBy === 'followers_count' ? 700 : 400,
                    cursor: 'pointer',
                    color: 'var(--ink)',
                  }}
                >
                  FOLLOWERS {sortBy === 'followers_count' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                </button>
              </div>
            </div>

            {/* Table Body */}
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <table className="forensic-table">
                <thead>
                  <tr>
                    <th>NODE IDX</th>
                    <th>SCREEN NAME</th>
                    <th>GROUND TRUTH</th>
                    <th>PREDICTED CLASS</th>
                    <th>BOT PROBABILITY</th>
                    <th>CONFIDENCE</th>
                    <th>FOLLOWERS</th>
                    <th>FOLLOWING</th>
                    <th>POSTS</th>
                  </tr>
                </thead>
                <tbody>
                  {tableLoading ? (
                    <tr>
                      <td colSpan={9} style={{ textAlign: 'center', padding: '24px', fontFamily: 'var(--font-mono)' }}>
                        SCANNING GRAPH DATABASE RECORDS...
                      </td>
                    </tr>
                  ) : accountsData.items.length === 0 ? (
                    <tr>
                      <td colSpan={9} style={{ textAlign: 'center', padding: '24px', fontFamily: 'var(--font-mono)' }}>
                        NO ACCOUNTS MATCHING SEARCH QUERY.
                      </td>
                    </tr>
                  ) : (
                    accountsData.items.map((acc) => {
                      const isSelected = selectedAccountId === acc.id;
                      const predClass = acc.prediction?.predicted_class || acc.ground_truth;
                      const pBot = acc.prediction?.bot_probability ?? 0.5;

                      return (
                        <tr
                          key={acc.id}
                          onClick={() => openDossierForAccount(acc.id)}
                          style={{
                            backgroundColor: isSelected ? 'var(--paper-hover)' : undefined,
                            fontWeight: isSelected ? 600 : 400,
                          }}
                        >
                          <td className="mono-data">#{acc.id}</td>
                          <td style={{ fontWeight: 600 }}>@{acc.screen_name}</td>
                          <td>
                            <span className={
                              acc.ground_truth === 'bot' ? 'badge-bot' : acc.ground_truth === 'human' ? 'badge-human' : 'badge-suspicious'
                            }>
                              {acc.ground_truth.toUpperCase()}
                            </span>
                          </td>
                          <td>
                            <span className={
                              predClass === 'bot' ? 'badge-bot' : predClass === 'human' ? 'badge-human' : 'badge-suspicious'
                            }>
                              {predClass.toUpperCase()}
                            </span>
                          </td>
                          <td className="mono-data">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span>{(pBot * 100).toFixed(1)}%</span>
                              <div style={{
                                width: '60px',
                                height: '6px',
                                backgroundColor: 'var(--rule-light)',
                                overflow: 'hidden',
                              }}>
                                <div style={{
                                  width: `${pBot * 100}%`,
                                  height: '100%',
                                  backgroundColor: pBot > 0.5 ? 'var(--stamp)' : 'var(--ledger)',
                                }} />
                              </div>
                            </div>
                          </td>
                          <td className="mono-data">{((acc.prediction?.confidence ?? 0.85) * 100).toFixed(1)}%</td>
                          <td className="mono-data">{acc.followers_count.toLocaleString()}</td>
                          <td className="mono-data">{acc.following_count.toLocaleString()}</td>
                          <td className="mono-data">{acc.post_count.toLocaleString()}</td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Table Pagination Bar */}
            <div style={{
              padding: '6px 16px',
              backgroundColor: 'var(--paper-deep)',
              borderTop: '1px solid var(--rule-light)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
            }}>
              <div>PAGE {page} OF {Math.max(1, Math.ceil(accountsData.total / 20))}</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '10px' }}
                >
                  PREVIOUS
                </button>
                <button
                  disabled={page * 20 >= accountsData.total}
                  onClick={() => setPage((p) => p + 1)}
                  className="btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '10px' }}
                >
                  NEXT
                </button>
              </div>
            </div>

          </div>

        </div>

        {/* Right Drawer: Account Dossier */}
        <AccountDossier />

      </div>

    </div>
  );
};
