const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${(import.meta.env.VITE_API_BASE_URL as string).replace(/\/$/, '')}/api/v1`
  : '/api/v1';

export interface DatasetSummary {
  id: number;
  name: string;
  description: string;
  record_count: number;
  edge_count: number;
  created_at: string;
  is_active: boolean;
  is_default: boolean;
  source: string;
  status: string;
  ingestion_stage?: string;
  ingestion_progress?: number;
  breakdown: {
    bot_count: number;
    human_count: number;
    suspicious_count: number;
  };
  density: number;
  community_count?: number;
  coordination_cluster_count?: number;
}

export interface AccountSummary {
  id: number;
  user_id_str: string;
  screen_name: string;
  name: string;
  followers_count: number;
  following_count: number;
  post_count: number;
  verified: boolean;
  ground_truth: string;
  dataset_id?: number;
  prediction?: {
    predicted_class: string;
    bot_probability: number;
    human_probability: number;
    suspicious_probability: number;
    confidence: number;
  };
}

export interface AccountDetail extends AccountSummary {
  description: string;
  location: string;
  created_at: string;
  account_age_days: number;
  listed_count: number;
  has_profile_image: boolean;
  has_description: boolean;
  default_profile: boolean;
  profile_completeness: number;
  account_features?: {
    follower_friend_ratio: number;
    rep_score: number;
    activity_rate: number;
    posts_per_day: number;
    reply_ratio: number;
    retweet_ratio: number;
    mention_ratio: number;
    url_ratio: number;
    hashtag_ratio: number;
    duplicate_content_ratio: number;
    tweet_hour_entropy: number;
    tweet_similarity_score: number;
    engagement_score: number;
    activity_consistency: number;
  };
  graph_features?: {
    in_degree: number;
    out_degree: number;
    total_degree: number;
    in_degree_centrality: number;
    out_degree_centrality: number;
    pagerank: number;
    betweenness_centrality: number;
    clustering_coeff: number;
    k_core: number;
  };
  community?: {
    community_id: number;
    role: string;
    classification: string;
    bot_concentration: number;
    size: number;
  };
  prediction?: {
    predicted_class: string;
    bot_probability: number;
    human_probability: number;
    suspicious_probability: number;
    confidence: number;
    latency_ms: number;
    signals: Array<{
      name: string;
      value: number;
      weight: number;
      category: string;
      description: string;
    }>;
  };
  recent_posts: Array<{
    id: number;
    tweet_id_str: string;
    text: string;
    timestamp: string;
    retweet_count: number;
    like_count: number;
    reply_count: number;
    has_url: boolean;
    has_hashtag: boolean;
  }>;
}

export interface GraphNode {
  id: string;
  label: string;
  followers: number;
  following: number;
  ground_truth: string;
  predicted_class: string;
  bot_probability: number;
  confidence: number;
  community_id: number;
  community_class: string;
  dataset_id?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation_type: string;
  weight: number;
  attention_weight?: number;
}

export interface SubgraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
  dataset_id?: number;
}

export interface CommunitySummary {
  id: number;
  dataset_id?: number;
  community_id: number;
  algorithm: string;
  size: number;
  n_bots: number;
  n_suspicious: number;
  n_humans: number;
  bot_concentration: number;
  suspicious_concentration: number;
  human_concentration: number;
  coordinated_ratio: number;
  internal_edges: number;
  internal_density: number;
  classification: string;
}

export interface CoordinationCluster {
  id: number;
  dataset_id?: number;
  cluster_name: string;
  coordination_type: string;
  account_count: number;
  avg_similarity: number;
  time_window_seconds: number;
  detected_at: string;
  status: string;
}

export interface ModelRun {
  id: number;
  model_name: string;
  architecture: string;
  accuracy: number;
  f1_macro: number;
  precision: number;
  recall: number;
  roc_auc: number;
  hyperparameters: Record<string, any>;
  checkpoint_path: string;
  is_active: boolean;
  created_at: string;
}

// Helper: build URL with optional dataset_id
function withDataset(base: string, datasetId: number | null | undefined, extra: Record<string, any> = {}): string {
  const q = new URLSearchParams();
  if (datasetId != null) q.append('dataset_id', String(datasetId));
  Object.entries(extra).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') q.append(k, String(v));
  });
  const qs = q.toString();
  return qs ? `${base}?${qs}` : base;
}

export const api = {
  // ── Datasets ─────────────────────────────────────────────────────────────
  getDatasets: async (): Promise<DatasetSummary[]> => {
    const res = await fetch(`${API_BASE}/datasets`);
    if (!res.ok) throw new Error('Failed to fetch datasets');
    return res.json();
  },

  getDataset: async (id: number): Promise<DatasetSummary> => {
    const res = await fetch(`${API_BASE}/datasets/${id}`);
    if (!res.ok) throw new Error('Failed to fetch dataset');
    return res.json();
  },

  getDatasetStatus: async (id: number) => {
    const res = await fetch(`${API_BASE}/datasets/${id}/status`);
    if (!res.ok) throw new Error('Failed to fetch dataset status');
    return res.json();
  },

  getRejectedRows: async (id: number) => {
    const res = await fetch(`${API_BASE}/datasets/${id}/rejected-rows`);
    if (!res.ok) throw new Error('Failed to fetch rejected rows');
    return res.json();
  },

  uploadDataset: async (file: File): Promise<{
    upload_id: number;
    dataset_id: number;
    column_suggestions: Record<string, string>;
    message: string;
  }> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  confirmDataset: async (uploadId: number, columnMap: Record<string, string> = {}): Promise<{
    dataset_id: number;
    job_id: string;
    rows_accepted: number;
    rows_rejected: number;
    warnings: string[];
    status: string;
  }> => {
    const form = new FormData();
    form.append('upload_id', String(uploadId));
    form.append('column_map', JSON.stringify(columnMap));
    const res = await fetch(`${API_BASE}/datasets/confirm`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Confirm failed' }));
      throw new Error(err.detail || 'Confirm failed');
    }
    return res.json();
  },

  deleteDataset: async (id: number) => {
    const res = await fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Delete failed' }));
      throw new Error(err.detail || 'Delete failed');
    }
    return res.json();
  },

  // ── Accounts ──────────────────────────────────────────────────────────────
  getAccounts: async (params: Record<string, any> = {}, datasetId?: number | null) => {
    const query = new URLSearchParams();
    if (datasetId != null) query.append('dataset_id', String(datasetId));
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') query.append(k, String(v));
    });
    const res = await fetch(`${API_BASE}/accounts?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch accounts');
    return res.json();
  },

  getAccountDetail: async (id: number): Promise<AccountDetail> => {
    const res = await fetch(`${API_BASE}/accounts/${id}`);
    if (!res.ok) throw new Error('Failed to fetch account detail');
    return res.json();
  },

  getDossierPdfUrl: (id: number) => `${API_BASE}/accounts/${id}/dossier/pdf`,

  // ── Graph ─────────────────────────────────────────────────────────────────
  getSubgraph: async (
    params: { limit_nodes?: number; community_id?: number; ground_truth?: string } = {},
    datasetId?: number | null,
  ): Promise<SubgraphResponse> => {
    const query = new URLSearchParams();
    if (datasetId != null) query.append('dataset_id', String(datasetId));
    if (params.limit_nodes) query.append('limit_nodes', String(params.limit_nodes));
    if (params.community_id !== undefined) query.append('community_id', String(params.community_id));
    if (params.ground_truth) query.append('ground_truth', params.ground_truth);
    const res = await fetch(`${API_BASE}/graph/subgraph?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch subgraph');
    return res.json();
  },

  getEgoGraph: async (accountId: number, hops = 1, maxNeighbors = 25) => {
    const res = await fetch(`${API_BASE}/graph/ego/${accountId}?hops=${hops}&max_neighbors=${maxNeighbors}`);
    if (!res.ok) throw new Error('Failed to fetch ego graph');
    return res.json();
  },

  // ── Prediction ────────────────────────────────────────────────────────────
  predictAccount: async (accountId: number, modelName = 'GAT') => {
    const res = await fetch(`${API_BASE}/predict/account/${accountId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    if (!res.ok) throw new Error('Inference failed');
    return res.json();
  },

  // ── Communities ───────────────────────────────────────────────────────────
  getCommunities: async (sortBy = 'bot_concentration', sortOrder = 'desc', datasetId?: number | null): Promise<CommunitySummary[]> => {
    const url = withDataset(`${API_BASE}/communities`, datasetId, { sort_by: sortBy, sort_order: sortOrder });
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch communities');
    return res.json();
  },

  getCommunityDetail: async (communityId: number, datasetId?: number | null) => {
    const url = withDataset(`${API_BASE}/communities/${communityId}`, datasetId);
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch community detail');
    return res.json();
  },

  // ── Coordination ──────────────────────────────────────────────────────────
  getCoordinationClusters: async (datasetId?: number | null): Promise<CoordinationCluster[]> => {
    const url = withDataset(`${API_BASE}/coordination/clusters`, datasetId);
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch coordination clusters');
    return res.json();
  },

  getCoordinationClusterDetail: async (id: number, datasetId?: number | null) => {
    const url = withDataset(`${API_BASE}/coordination/clusters/${id}`, datasetId);
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch coordination cluster detail');
    return res.json();
  },

  // ── Temporal ──────────────────────────────────────────────────────────────
  getTemporalAnalysis: async () => {
    const res = await fetch(`${API_BASE}/temporal/bursts`);
    if (!res.ok) throw new Error('Failed to fetch temporal data');
    return res.json();
  },

  // ── Models ────────────────────────────────────────────────────────────────
  getModels: async (): Promise<ModelRun[]> => {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error('Failed to fetch model registry');
    return res.json();
  },

  activateModel: async (id: number) => {
    const res = await fetch(`${API_BASE}/models/${id}/activate`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to activate model');
    return res.json();
  },

  // ── Admin ─────────────────────────────────────────────────────────────────
  getAdminStats: async () => {
    const res = await fetch(`${API_BASE}/admin/stats`);
    if (!res.ok) throw new Error('Failed to fetch system stats');
    return res.json();
  },

  getAuditLogs: async (page = 1, pageSize = 25) => {
    const res = await fetch(`${API_BASE}/admin/audit-logs?page=${page}&page_size=${pageSize}`);
    if (!res.ok) throw new Error('Failed to fetch audit logs');
    return res.json();
  },
};
