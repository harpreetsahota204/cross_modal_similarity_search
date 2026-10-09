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
  uri: string;
  indexes: IndexInfo[];
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

export type QueryMode = "text" | "samples" | "file";
export type RunSection = {
  key: string;
  modality: Modality;
  count: number;
  top_score: number | null;
  error: string | null;
};

/** A saved search. ``query`` is the text, the sample IDs or the filename. */
export type Run = {
  run_id: string;
  run_name: string;
  query_type: QueryMode;
  query: string | string[] | null;
  query_label: string | null;
  modality: Modality | null;
  brain_keys: string[];
  k: number;
  status: "completed" | "failed";
  error: string | null;
  sections: RunSection[];
  result_count: number;
  creation_time: string;
};

export type Applied = { run_id: string; key: string };

export type PanelData = {
  status?: Status;
  runs?: Run[];
  selection?: Selection | null;
  model_loaded?: boolean;
  applied?: Applied | null;
};

/** Python panel methods, as event names the App can trigger. */
export type PanelMethods = {
  refresh: string;
  search: string;
  apply_run: string;
  delete_run: string;
  clear_view: string;
  build_index: string;
  rebuild_table: string;
};

export type UploadedQuery = {
  name: string;
  data: string;
  kind: MediaKind;
};

/** What the New Search form starts from when cloning a search. */
export type FormSeed = {
  mode: QueryMode;
  text: string;
  modality: Modality | null;
  brainKeys: string[];
  k: number;
};

export type Page = "home" | "new_search" | "indexes";
