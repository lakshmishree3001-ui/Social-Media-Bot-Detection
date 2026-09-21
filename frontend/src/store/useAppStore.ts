import { create } from 'zustand';

export interface DatasetMeta {
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
}

interface AppState {
  selectedAccountId: number | null;
  activeModel: string;
  searchQuery: string;
  selectedCommunityId: number | null;
  filterGroundTruth: string | null;
  filterPredictedClass: string | null;
  isDossierOpen: boolean;

  // Multi-dataset state
  selectedDatasetId: number | null;
  datasets: DatasetMeta[];

  setSelectedAccountId: (id: number | null) => void;
  setActiveModel: (model: string) => void;
  setSearchQuery: (query: string) => void;
  setSelectedCommunityId: (id: number | null) => void;
  setFilterGroundTruth: (gt: string | null) => void;
  setFilterPredictedClass: (pred: string | null) => void;
  setIsDossierOpen: (open: boolean) => void;
  openDossierForAccount: (id: number) => void;

  setSelectedDatasetId: (id: number | null) => void;
  setDatasets: (datasets: DatasetMeta[]) => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedAccountId: 0,
  activeModel: 'GAT',
  searchQuery: '',
  selectedCommunityId: null,
  filterGroundTruth: null,
  filterPredictedClass: null,
  isDossierOpen: true,

  selectedDatasetId: null,  // null = use server default (is_default=true)
  datasets: [],

  setSelectedAccountId: (id) => set({ selectedAccountId: id }),
  setActiveModel: (model) => set({ activeModel: model }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setSelectedCommunityId: (id) => set({ selectedCommunityId: id }),
  setFilterGroundTruth: (gt) => set({ filterGroundTruth: gt }),
  setFilterPredictedClass: (pred) => set({ filterPredictedClass: pred }),
  setIsDossierOpen: (open) => set({ isDossierOpen: open }),
  openDossierForAccount: (id) => set({ selectedAccountId: id, isDossierOpen: true }),

  setSelectedDatasetId: (id) => set({ selectedDatasetId: id }),
  setDatasets: (datasets) => set({ datasets }),
}));

