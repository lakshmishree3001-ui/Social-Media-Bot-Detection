import React, { useEffect, useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { api, DatasetSummary } from '../api/client';

type UploadStep = 'idle' | 'uploading' | 'mapping' | 'confirming' | 'ingesting' | 'done' | 'error';

interface UploadState {
  step: UploadStep;
  uploadId?: number;
  datasetId?: number;
  jobId?: string;
  columnSuggestions?: Record<string, string>;
  columnMap: Record<string, string>;
  rowsAccepted?: number;
  rowsRejected?: number;
  warnings?: string[];
  error?: string;
  progress?: number;
  stage?: string;
}

function StatusBadge({ status }: { status: string }) {
  const isReady = status === 'ready';
  const isIngesting = status === 'ingesting';
  return (
    <span style={{
      fontFamily: 'var(--font-mono)',
      fontSize: '10px',
      fontWeight: 700,
      letterSpacing: '0.06em',
      padding: '2px 6px',
      border: isReady ? '1px solid var(--ledger)' : isIngesting ? '1px solid var(--signal)' : '1px solid var(--stamp)',
      color: isReady ? 'var(--ledger)' : isIngesting ? 'var(--signal-text)' : 'var(--stamp)',
      backgroundColor: isReady ? 'var(--ledger-bg)' : 'var(--paper-deep)',
    }}>
      [{status.toUpperCase()}]
    </span>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div style={{ height: 4, background: 'var(--rule-light)', width: '100%', marginTop: 6 }}>
      <div style={{
        height: 4, width: `${Math.round(value * 100)}%`,
        background: 'var(--stamp)', transition: 'width 400ms ease',
      }} />
    </div>
  );
}

export const DatasetManagementPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadState, setUploadState] = useState<UploadState>({ step: 'idle', columnMap: {} });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rejectedRows, setRejectedRows] = useState<any[]>([]);
  const [showRejected, setShowRejected] = useState<number | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDatasets = () => {
    api.getDatasets()
      .then(setDatasets)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadDatasets();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  // Auto-poll datasets if any dataset is currently ingesting
  useEffect(() => {
    const hasIngesting = datasets.some(d => d.status === 'ingesting');
    if (hasIngesting) {
      const timer = setInterval(() => {
        loadDatasets();
      }, 2500);
      return () => clearInterval(timer);
    }
  }, [datasets]);

  // Poll ingestion status
  useEffect(() => {
    if (uploadState.step === 'ingesting' && uploadState.datasetId) {
      const dsId = uploadState.datasetId;
      pollRef.current = setInterval(async () => {
        try {
          const s = await api.getDatasetStatus(dsId);
          setUploadState(prev => ({
            ...prev,
            progress: s.progress,
            stage: s.stage,
          }));
          if (s.status === 'ready') {
            setUploadState(prev => ({ ...prev, step: 'done' }));
            if (pollRef.current) clearInterval(pollRef.current);
            loadDatasets();
          } else if (s.status === 'failed') {
            setUploadState(prev => ({ ...prev, step: 'error', error: s.error || 'Pipeline failed' }));
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch (_) {}
      }, 2500);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [uploadState.step, uploadState.datasetId]);

  // Auto-poll datasets list whenever any loaded dataset is 'ingesting'
  useEffect(() => {
    const hasIngesting = datasets.some(d => d.status === 'ingesting');
    if (!hasIngesting) return;

    const interval = setInterval(() => {
      api.getDatasets()
        .then(setDatasets)
        .catch(() => {});
    }, 2500);

    return () => clearInterval(interval);
  }, [datasets]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setSelectedFile(f);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploadState({ step: 'uploading', columnMap: {} });
    try {
      const res = await api.uploadDataset(selectedFile);
      setUploadState({
        step: 'mapping',
        uploadId: res.upload_id,
        datasetId: res.dataset_id,
        columnSuggestions: res.column_suggestions,
        columnMap: { ...res.column_suggestions },
      });
    } catch (e: any) {
      setUploadState({ step: 'error', error: e.message, columnMap: {} });
    }
  };

  const handleConfirm = async () => {
    if (!uploadState.uploadId) return;
    setUploadState(prev => ({ ...prev, step: 'confirming' }));
    try {
      const res = await api.confirmDataset(uploadState.uploadId, uploadState.columnMap);
      setUploadState(prev => ({
        ...prev,
        step: 'ingesting',
        jobId: res.job_id,
        rowsAccepted: res.rows_accepted,
        rowsRejected: res.rows_rejected,
        warnings: res.warnings,
        progress: 0,
        stage: 'queued',
      }));
      loadDatasets();
    } catch (e: any) {
      setUploadState(prev => ({ ...prev, step: 'error', error: e.message }));
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this dataset and all its scoped data? This cannot be undone.')) return;
    try {
      await api.deleteDataset(id);
      loadDatasets();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleShowRejected = async (id: number) => {
    try {
      const r = await api.getRejectedRows(id);
      setRejectedRows(r.rows || []);
      setShowRejected(id);
    } catch (_) {}
  };

  const updateColMap = (rawCol: string, canonicalCol: string) => {
    setUploadState(prev => ({
      ...prev,
      columnMap: { ...prev.columnMap, [rawCol]: canonicalCol },
    }));
  };

  const mono: React.CSSProperties = {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    letterSpacing: '0.04em',
  };

  const label: React.CSSProperties = {
    ...mono,
    color: 'var(--ink-muted)',
    textTransform: 'uppercase',
    display: 'block',
    marginBottom: 4,
  };

  const value: React.CSSProperties = {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--ink)',
    fontWeight: 600,
  };

  const btn = (variant: 'primary' | 'ghost' | 'danger'): React.CSSProperties => ({
    padding: '7px 16px',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    letterSpacing: '0.05em',
    border: variant === 'ghost' ? '1px solid var(--rule)' : variant === 'danger' ? '1px solid #a00000' : '1px solid var(--ink)',
    background: variant === 'primary' ? 'var(--ink)' : variant === 'danger' ? '#a00000' : 'transparent',
    color: variant === 'primary' || variant === 'danger' ? 'var(--paper)' : 'var(--ink)',
    cursor: 'pointer',
    textTransform: 'uppercase',
  });

  const card: React.CSSProperties = {
    border: '1px solid var(--rule)',
    padding: '20px 24px',
    marginBottom: 16,
    background: 'var(--paper)',
  };

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '40px 24px 80px' }}>
      {/* Page header */}
      <div style={{ marginBottom: 32, borderBottom: '1.5px solid var(--ink)', paddingBottom: 16 }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase', margin: 0 }}>
          Dataset Management
        </h1>
        <p style={{ ...mono, color: 'var(--ink-muted)', marginTop: 6 }}>
          Upload a CSV/TSV, confirm column mapping, and run the full 8-stage ingestion pipeline.
          The seed dataset cannot be deleted.
        </p>
      </div>

      {/* Existing datasets */}
      <section style={{ marginBottom: 48 }}>
        <h2 style={{ ...mono, fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16, borderBottom: '1px solid var(--rule-light)', paddingBottom: 8 }}>
          Loaded Datasets
        </h2>
        {loading && <p style={{ ...mono, color: 'var(--ink-muted)' }}>Loading...</p>}
        {!loading && datasets.length === 0 && (
          <p style={{ ...mono, color: 'var(--ink-muted)' }}>No datasets found.</p>
        )}
        {datasets.map(d => (
          <div key={d.id} style={{
            borderBottom: '1px solid var(--rule)',
            padding: '20px 0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: 16,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <StatusBadge status={d.status} />
                <span style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 700 }}>{d.name}</span>
                {d.is_default && (
                  <span style={{ ...mono, fontSize: 9, padding: '1px 6px', border: '1px solid var(--ink)', letterSpacing: '0.06em' }}>
                    SEED / DEFAULT
                  </span>
                )}
              </div>
              {d.description && (
                <p style={{ ...mono, fontSize: 10.5, color: 'var(--ink-muted)', margin: '0 0 12px 0' }}>{d.description}</p>
              )}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, auto)', gap: '0 28px' }}>
                {[
                  ['ID', `#${d.id}`],
                  ['Accounts', d.record_count.toLocaleString()],
                  ['Edges', d.edge_count.toLocaleString()],
                  ['Bots', (d.breakdown?.bot_count ?? 0).toLocaleString()],
                  ['Source', d.source?.toUpperCase()],
                ].map(([k, v]) => (
                  <div key={k as string}>
                    <span style={label}>{k}</span>
                    <span style={value}>{v}</span>
                  </div>
                ))}
              </div>
              {d.status === 'ingesting' && (
                <ProgressBar value={d.ingestion_progress ?? 0} />
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
              <Link
                to={`/app?dataset_id=${d.id}`}
                style={{ ...btn('ghost'), textDecoration: 'none', display: 'inline-block' }}
              >
                Open
              </Link>
              <button
                style={btn('ghost')}
                onClick={() => handleShowRejected(d.id)}
              >
                Rejected Rows
              </button>
              {!d.is_default && (
                <button style={btn('danger')} onClick={() => handleDelete(d.id)}>
                  Delete
                </button>
              )}
            </div>
          </div>
        ))}
      </section>

      {/* Rejected rows panel */}
      {showRejected !== null && (
        <div style={{ ...card, marginBottom: 32, background: 'var(--paper-deep)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ ...mono, fontWeight: 600 }}>Rejected Rows — Dataset #{showRejected}</span>
            <button style={btn('ghost')} onClick={() => setShowRejected(null)}>Close</button>
          </div>
          {rejectedRows.length === 0 ? (
            <p style={{ ...mono, color: 'var(--ink-muted)' }}>No rejected rows.</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--rule)' }}>
                  {['Row Index', 'Screen Name / Preview', 'Reason'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--ink-muted)', letterSpacing: '0.05em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rejectedRows.slice(0, 100).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--rule-light)' }}>
                    <td style={{ padding: '4px 8px', color: 'var(--ink-muted)' }}>{r.row_index}</td>
                    <td style={{ padding: '4px 8px' }}>{r.row_preview}</td>
                    <td style={{ padding: '4px 8px', color: '#a00000' }}>{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Upload panel */}
      <section>
        <h2 style={{ ...mono, fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16, borderBottom: '1px solid var(--rule-light)', paddingBottom: 8 }}>
          Upload New Dataset
        </h2>

        {/* Step: idle / file selection */}
        {(uploadState.step === 'idle' || uploadState.step === 'error') && (
          <div style={card}>
            <p style={{ ...mono, color: 'var(--ink-muted)', marginBottom: 16 }}>
              Accepted formats: CSV, TSV. Maximum size: 200 MB.
              Required columns: <strong>screen_name, followers_count, following_count, post_count, account_age_days</strong>.
            </p>
            {uploadState.step === 'error' && (
              <div style={{ padding: '8px 12px', border: '1px solid #a00000', color: '#a00000', ...mono, fontSize: 11, marginBottom: 12 }}>
                {uploadState.error}
              </div>
            )}
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.txt"
                onChange={handleFileChange}
                style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink)' }}
                id="dataset-file-input"
              />
              <button
                id="dataset-upload-btn"
                style={btn('primary')}
                disabled={!selectedFile}
                onClick={handleUpload}
              >
                Upload
              </button>
            </div>
          </div>
        )}

        {/* Step: uploading */}
        {uploadState.step === 'uploading' && (
          <div style={card}>
            <p style={{ ...mono, color: 'var(--ink-muted)' }}>Uploading and parsing file headers...</p>
            <ProgressBar value={0.3} />
          </div>
        )}

        {/* Step: column mapping */}
        {uploadState.step === 'mapping' && (
          <div style={card}>
            <h3 style={{ ...mono, fontSize: 12, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 12 }}>
              Column Mapping — Dataset #{uploadState.datasetId}
            </h3>
            <p style={{ ...mono, color: 'var(--ink-muted)', marginBottom: 16 }}>
              Map your CSV columns to canonical field names. Only non-trivial mappings are shown below.
              Columns already matching canonical names are automatically accepted.
            </p>
            {Object.keys(uploadState.columnSuggestions || {}).length === 0 ? (
              <p style={{ ...mono, color: 'var(--ink-muted)', marginBottom: 16 }}>
                All required columns detected automatically. No mapping required.
              </p>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 20 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--rule)' }}>
                    {['Raw Column (in your file)', 'Maps to Canonical Column'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-muted)', letterSpacing: '0.05em' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(uploadState.columnMap).map(([raw, canonical]) => (
                    <tr key={raw} style={{ borderBottom: '1px solid var(--rule-light)' }}>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: 11 }}>{raw}</td>
                      <td style={{ padding: '6px 8px' }}>
                        <input
                          value={canonical}
                          onChange={e => updateColMap(raw, e.target.value)}
                          style={{
                            fontFamily: 'var(--font-mono)', fontSize: 11,
                            border: '1px solid var(--rule)', padding: '3px 8px',
                            background: 'var(--paper-deep)', color: 'var(--ink)', width: '100%',
                          }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <div style={{ display: 'flex', gap: 12 }}>
              <button id="dataset-confirm-btn" style={btn('primary')} onClick={handleConfirm}>
                Confirm and Start Ingestion
              </button>
              <button style={btn('ghost')} onClick={() => setUploadState({ step: 'idle', columnMap: {} })}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Step: confirming */}
        {uploadState.step === 'confirming' && (
          <div style={card}>
            <p style={{ ...mono, color: 'var(--ink-muted)' }}>Validating all rows and queuing ingestion pipeline...</p>
            <ProgressBar value={0.1} />
          </div>
        )}

        {/* Step: ingesting */}
        {uploadState.step === 'ingesting' && (
          <div style={card}>
            <h3 style={{ ...mono, fontSize: 12, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 8 }}>
              Pipeline Running — Dataset #{uploadState.datasetId}
            </h3>
            {uploadState.warnings && uploadState.warnings.length > 0 && (
              <div style={{ padding: '6px 10px', border: '1px solid #c8a000', color: '#c8a000', ...mono, fontSize: 10, marginBottom: 10 }}>
                {uploadState.warnings.join(' | ')}
              </div>
            )}
            <div style={{ ...mono, color: 'var(--ink-muted)', marginBottom: 6 }}>
              Stage: <strong style={{ color: 'var(--ink)' }}>{(uploadState.stage || 'queued').toUpperCase()}</strong>
              &nbsp;&nbsp;Rows accepted: <strong style={{ color: 'var(--ink)' }}>{uploadState.rowsAccepted?.toLocaleString()}</strong>
              &nbsp;&nbsp;Rejected: <strong style={{ color: '#c8a000' }}>{uploadState.rowsRejected}</strong>
            </div>
            <ProgressBar value={uploadState.progress ?? 0} />
            <p style={{ ...mono, fontSize: 10, color: 'var(--ink-muted)', marginTop: 8 }}>
              Stages: Normalise → Graph → Community Detection → Coordination → GNN Inference → Finalise
            </p>
          </div>
        )}

        {/* Step: done */}
        {uploadState.step === 'done' && (
          <div style={{ ...card, border: '1px solid var(--ledger)' }}>
            <p style={{ ...mono, color: 'var(--ledger)', marginBottom: 12 }}>
              Ingestion complete. Dataset #{uploadState.datasetId} is now ready for analysis.
            </p>
            <div style={{ display: 'flex', gap: 12 }}>
              <Link
                to={`/app?dataset_id=${uploadState.datasetId}`}
                style={{ ...btn('primary'), textDecoration: 'none', display: 'inline-block' }}
              >
                Open in Analyst Workbench
              </Link>
              <button style={btn('ghost')} onClick={() => setUploadState({ step: 'idle', columnMap: {} })}>
                Upload Another
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
