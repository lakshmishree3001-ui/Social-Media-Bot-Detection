import React, { useEffect, useState } from 'react';
import { useAppStore, DatasetMeta } from '../../store/useAppStore';
import { api } from '../../api/client';

export const DatasetSwitcher: React.FC = () => {
  const { selectedDatasetId, datasets, setSelectedDatasetId, setDatasets } = useAppStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.getDatasets()
      .then((data) => {
        setDatasets(data as DatasetMeta[]);
        // Auto-select the default if nothing is selected yet
        if (selectedDatasetId === null) {
          const def = data.find((d: any) => d.is_default);
          if (def) setSelectedDatasetId(def.id);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const current = datasets.find(d => d.id === selectedDatasetId) || datasets[0];

  return (
    <div style={{ position: 'relative' }}>
      <button
        id="dataset-switcher-btn"
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          padding: '4px 12px',
          minWidth: '240px',
          maxWidth: '340px',
          background: 'var(--paper-deep)',
          border: '1.5px solid var(--ink)',
          color: 'var(--ink)',
          cursor: 'pointer',
          textAlign: 'left',
          letterSpacing: '0.03em',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', overflow: 'hidden' }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 700,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: 'var(--ink)',
          }}>
            {loading ? 'LOADING DATASET...' : (current?.name?.toUpperCase() || 'SELECT DATASET')}
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '9.5px',
            color: 'var(--ink-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <span>
              {current?.record_count ? current.record_count.toLocaleString() : (current?.id === 1 ? '10,000' : '0')} ACCOUNTS
              {' / '}
              {current?.edge_count ? current.edge_count.toLocaleString() : (current?.id === 1 ? '54,980' : '0')} EDGES
            </span>
            <span>·</span>
            <strong style={{ color: current?.status === 'ready' ? 'var(--ledger)' : 'var(--stamp)' }}>
              {current?.status?.toUpperCase() || 'READY'}
            </strong>
          </div>
        </div>
        <span style={{ color: 'var(--ink)', fontSize: '10px', flexShrink: 0 }}>
          {open ? '▲' : '▼'}
        </span>
      </button>

      {open && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 4px)',
          right: 0,
          minWidth: '280px',
          backgroundColor: 'var(--paper)',
          border: '1.5px solid var(--ink)',
          zIndex: 500,
          boxShadow: 'none',
        }}>
          {datasets.length === 0 && (
            <div style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)' }}>
              No datasets available.
            </div>
          )}
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => { setSelectedDatasetId(d.id); setOpen(false); }}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                padding: '10px 16px',
                background: d.id === selectedDatasetId ? 'var(--ink)' : 'transparent',
                color: d.id === selectedDatasetId ? 'var(--paper)' : 'var(--ink)',
                border: 'none',
                borderBottom: '1px solid var(--rule-light)',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                letterSpacing: '0.03em',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600 }}>{d.name}</span>
                {d.is_default && (
                  <span style={{
                    fontSize: '9px', padding: '1px 5px',
                    background: d.id === selectedDatasetId ? 'var(--paper)' : 'var(--ink)',
                    color: d.id === selectedDatasetId ? 'var(--ink)' : 'var(--paper)',
                    letterSpacing: '0.05em',
                  }}>SEED</span>
                )}
              </div>
              <div style={{
                fontSize: '10px',
                color: d.id === selectedDatasetId ? 'rgba(242,237,227,0.7)' : 'var(--ink-muted)',
                marginTop: '2px',
              }}>
                {d.record_count.toLocaleString()} accounts &middot; {d.status.toUpperCase()}
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Click-away */}
      {open && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 499 }}
          onClick={() => setOpen(false)}
        />
      )}
    </div>
  );
};
