import { Size, Spinner } from "@voxel51/voodo";
import React, { useCallback } from "react";
import Home from "./components/Home";
import Indexes from "./components/Indexes";
import NewSearch from "./components/NewSearch";
import Setup from "./components/Setup";
import { usePanelClient } from "./hooks/usePanelClient";
import { useForgetPersistentState, usePersistentState } from "./hooks/usePersistentState";
import type { FormSeed, Page, PanelData, PanelMethods, Run } from "./types";

type Props = {
  data: PanelData;
  schema: { view: PanelMethods & Record<string, unknown> };
};

/** Pages laid out like FiftyOne's Similarity Search panel: saved searches,
 * a New Search form, and the indexes. */
export default function CrossModalRetrievalView({ data, schema }: Props) {
  const call = usePanelClient(schema.view);
  const forget = useForgetPersistentState();
  const [page, setPage] = usePersistentState<Page>("page", "home");
  const [seed, setSeed] = usePersistentState<FormSeed | null>("seed", null);

  const status = data?.status;
  const indexes = status?.indexes ?? [];
  const runs = data?.runs ?? [];

  const openForm = useCallback(
    (next: FormSeed | null) => {
      forget("form.");
      setSeed(next);
      setPage("new_search");
    },
    [forget, setSeed, setPage],
  );

  const onClone = useCallback(
    (run: Run) =>
      openForm({
        mode: run.query_type,
        text: run.query_type === "text" ? String(run.query ?? "") : "",
        modality: run.modality,
        brainKeys: run.brain_keys.length ? run.brain_keys : run.sections.map((s) => s.key),
        k: run.k,
      }),
    [openForm],
  );

  const onApply = useCallback((runId: string, key?: string) => call("apply_run", { run_id: runId, key }), [call]);
  const onDelete = useCallback((runId: string) => call("delete_run", { run_id: runId }), [call]);
  const onBuild = useCallback(() => call("build_index"), [call]);
  const onRebuild = useCallback((key: string) => call("rebuild_table", { key }), [call]);

  const onSearch = useCallback(
    async (params: Record<string, unknown>) => {
      await call("search", params);
      setPage("home");
    },
    [call, setPage],
  );

  if (!status) {
    return (
      <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Spinner size={Size.Md} />
      </div>
    );
  }

  if (indexes.length === 0 && runs.length === 0) {
    return <Setup uri={status.uri} onBuild={onBuild} />;
  }

  if (page === "new_search") {
    return (
      <NewSearch
        indexes={indexes}
        selection={data?.selection ?? null}
        seed={seed}
        modelLoaded={!!data?.model_loaded}
        onBack={() => setPage("home")}
        onSearch={onSearch}
      />
    );
  }

  if (page === "indexes") {
    return (
      <Indexes
        indexes={indexes}
        uri={status.uri}
        onBack={() => setPage("home")}
        onBuild={onBuild}
        onRebuild={onRebuild}
      />
    );
  }

  return (
    <Home
      runs={runs}
      hasIndexes={indexes.some((i) => i.ok)}
      applied={data?.applied ?? null}
      onApply={onApply}
      onClone={onClone}
      onDelete={onDelete}
      onRefresh={() => call("refresh")}
      onNewSearch={() => openForm(null)}
      onIndexes={() => setPage("indexes")}
    />
  );
}
