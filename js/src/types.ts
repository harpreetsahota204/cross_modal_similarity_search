export type Modality = "image" | "video" | "audio" | "video+audio";

export type MediaKind = "image" | "video" | "audio" | "mcap" | null;

export type IndexInfo = {
  key: string;
  modality: Modality;
  field: string;
  dim: number;
  prompt_name: string | null;
  table: string | null;
  count: number;
  ok: boolean;
  error: string | null;
  slices: string[] | null;
};

export type Status = {
  dataset: string;
  uri: string;
  indexes: IndexInfo[];
  label_fields: string[];
  default_label_field: string | null;
};

export type Selection = {
  ids: string[];
  count: number;
  kind: MediaKind;
  modalities: Modality[];
  /** modality -> brain key of an index that already stores the vectors */
  stored: Partial<Record<Modality, string>>;
  error: string | null;
};

export type Hit = {
  id: string;
  score: number;
  filepath: string;
  media: MediaKind;
  label: string | null;
  group_id?: string;
  slice?: string;
};

export type Section = {
  key: string;
  modality: Modality;
  hits: Hit[];
  error: string | null;
};

export type Results = {
  query: string;
  sections: Section[];
};

export type PanelData = {
  status?: Status;
  selection?: Selection | null;
  results?: Results | null;
  error?: string | null;
  model_loaded?: boolean;
  grid?: { key: string; count?: number } | null;
};

/** Python panel methods, as event names the App can trigger. */
export type PanelMethods = {
  refresh: string;
  search: string;
  show_in_grid: string;
  clear_grid: string;
  open_sample: string;
  build_index: string;
  rebuild_table: string;
};

export type QueryMode = "text" | "samples" | "file";

export type UploadedQuery = {
  name: string;
  data: string;
  kind: MediaKind;
};
