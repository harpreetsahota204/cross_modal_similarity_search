import {
  Align,
  Button,
  FormField,
  IconAction,
  IconName,
  Input,
  InputType,
  Justify,
  Orientation,
  Select,
  Size,
  Spacing,
  Spinner,
  Stack,
  Text,
  TextColor,
  TextVariant,
  Tooltip,
  Variant,
} from "@voxel51/voodo";
import React, { useCallback, useState } from "react";
import IndexPicker from "./components/IndexPicker";
import QueryBar from "./components/QueryBar";
import ResultStrip from "./components/ResultStrip";
import Setup from "./components/Setup";
import { BORDER, QUERY_MODALITIES } from "./format";
import { usePanelClient } from "./hooks/usePanelClient";
import { usePersistentState } from "./hooks/usePersistentState";
import type { Hit, Modality, PanelData, PanelMethods, QueryMode, UploadedQuery } from "./types";

type Props = {
  data: PanelData;
  schema: { view: PanelMethods & Record<string, unknown> };
};

const NO_CAPTION = "__none__";

export default function CrossModalRetrievalView({ data, schema }: Props) {
  const call = usePanelClient(schema.view);
  const status = data?.status;
  const indexes = status?.indexes ?? [];
  const selection = data?.selection ?? null;
  const results = data?.results ?? null;

  const [mode, setMode] = usePersistentState<QueryMode>("mode", "text");
  const [text, setText] = usePersistentState("text", "");
  const [upload, setUpload] = usePersistentState<UploadedQuery | null>("upload", null);
  const [modality, setModality] = usePersistentState<Modality | null>("modality", null);
  // Disabled rather than enabled keys, so new indexes are searched by default
  const [disabled, setDisabled] = usePersistentState<string[]>("disabled", []);
  const [k, setK] = usePersistentState("k", 12);
  const [caption, setCaption] = usePersistentState<string | null | undefined>("caption", undefined);
  const [searching, setSearching] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const enabled = indexes.filter((i) => i.ok && !disabled.includes(i.key)).map((i) => i.key);
  const labelField = caption === undefined ? status?.default_label_field ?? null : caption;

  // The modality buttons offer what the current query's media allows
  const options: Modality[] =
    mode === "samples"
      ? selection?.modalities ?? []
      : mode === "file" && upload?.kind
        ? QUERY_MODALITIES[upload.kind]
        : [];
  const preferred = mode === "samples" ? options.find((m) => selection?.stored[m]) : undefined;
  const activeModality = modality && options.includes(modality) ? modality : preferred ?? options[0] ?? null;

  const canSearch =
    enabled.length > 0 &&
    (mode === "text"
      ? text.trim().length > 0
      : mode === "samples"
        ? !!selection && !selection.error && !!activeModality
        : !!upload && !!activeModality);

  const runSearch = useCallback(
    async (params: Record<string, unknown>) => {
      setSearching(true);
      setLocalError(null);
      try {
        await call("search", { ...params, brain_keys: enabled, k, label_field: labelField });
      } finally {
        setSearching(false);
      }
    },
    [call, enabled, k, labelField],
  );

  const onSearch = () => {
    if (mode === "text") runSearch({ mode, text });
    else if (mode === "samples") runSearch({ mode, sample_ids: selection?.ids ?? [], modality: activeModality });
    else runSearch({ mode, file: upload, modality: activeModality });
  };

  const onPivot = (hit: Hit, m: Modality) => runSearch({ mode: "samples", sample_ids: [hit.id], modality: m });
  const onOpen = (hit: Hit) => call("open_sample", { id: hit.id, group_id: hit.group_id, slice: hit.slice });
  const onBuild = () => call("build_index");

  if (status && indexes.length === 0) {
    return (
      <div style={{ height: "100%", overflow: "auto" }}>
        <Setup uri={status.uri} onBuild={onBuild} />
      </div>
    );
  }

  const error = localError ?? data?.error ?? null;
  const slicesOf = (key: string) => indexes.find((i) => i.key === key)?.slices ?? null;

  return (
    <div style={{ height: "100%", overflow: "auto", boxSizing: "border-box" }}>
      <div style={{ padding: "12px 14px", borderBottom: `1px solid ${BORDER}` }}>
        <Stack orientation={Orientation.Column} spacing={Spacing.Md}>
          <Stack orientation={Orientation.Row} align={Align.Center} justify={Justify.Between}>
            <Stack orientation={Orientation.Column}>
              <Text variant={TextVariant.HeadingSm}>Cross-Modal Retrieval</Text>
              <Text variant={TextVariant.Caption} color={TextColor.Muted}>
                EmbeddingGemma 2 · LanceDB at {status?.uri ?? "…"}
              </Text>
            </Stack>
            <Stack orientation={Orientation.Row} spacing={Spacing.Xs}>
              <Button size={Size.Sm} variant={Variant.Secondary} leadingIcon={IconName.Add} onClick={onBuild}>
                New index
              </Button>
              <Tooltip content="Reload indexes and selection">
                <IconAction
                  size={Size.Sm}
                  icon={IconName.Refresh}
                  aria-label="Reload indexes and selection"
                  onClick={() => call("refresh")}
                />
              </Tooltip>
            </Stack>
          </Stack>

          <QueryBar
            mode={mode}
            onMode={setMode}
            text={text}
            onText={setText}
            selection={selection}
            upload={upload}
            onUpload={setUpload}
            modality={activeModality}
            onModality={setModality}
            canSearch={canSearch}
            searching={searching}
            onSearch={onSearch}
            onError={setLocalError}
          />

          <IndexPicker
            indexes={indexes}
            enabled={enabled}
            onEnabled={(keys) => setDisabled(indexes.map((i) => i.key).filter((key) => !keys.includes(key)))}
            onRebuild={(key) => call("rebuild_table", { key })}
          />

          <Stack orientation={Orientation.Row} spacing={Spacing.Lg} align={Align.End}>
            <div style={{ width: 120 }}>
              <FormField
                label="Results per index"
                control={
                  <Input
                    size={Size.Sm}
                    type={InputType.Number}
                    value={String(k)}
                    min={1}
                    max={100}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setK(Math.max(1, Math.min(100, Number(e.target.value) || 1)))
                    }
                  />
                }
              />
            </div>
            <div style={{ width: 200 }}>
              <FormField
                label="Caption"
                control={
                  <Select
                    exclusive
                    value={labelField ?? NO_CAPTION}
                    onChange={(v) => setCaption(v === NO_CAPTION ? null : (v as string))}
                    options={[
                      { id: NO_CAPTION, data: { label: "Filename" } },
                      ...(status?.label_fields ?? []).map((f) => ({ id: f, data: { label: f } })),
                    ]}
                  />
                }
              />
            </div>
          </Stack>
        </Stack>
      </div>

      <div style={{ padding: "12px 14px" }}>
        <Stack orientation={Orientation.Column} spacing={Spacing.Lg}>
          {searching && (
            <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center}>
              <Spinner size={Size.Sm} />
              <Text variant={TextVariant.BodyTertiary} color={TextColor.Secondary}>
                {data?.model_loaded
                  ? "Searching…"
                  : "Loading EmbeddingGemma 2 on the server; only the first search waits for this (about 15 s)"}
              </Text>
            </Stack>
          )}

          {error && (
            <Text variant={TextVariant.BodyTertiary} color={TextColor.Failure}>
              {error}
            </Text>
          )}

          {results && (
            <Stack orientation={Orientation.Row} align={Align.Center} justify={Justify.Between}>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Results for {results.query}
              </Text>
              {data?.grid && (
                <Button size={Size.Xs} variant={Variant.Secondary} leadingIcon={IconName.Close} onClick={() => call("clear_grid")}>
                  Restore grid
                </Button>
              )}
            </Stack>
          )}

          {results?.sections.map((section) => (
            <ResultStrip
              key={section.key}
              section={section}
              slices={slicesOf(section.key)}
              onShowInGrid={(ids) => call("show_in_grid", { key: section.key, ids })}
              onOpen={onOpen}
              onPivot={onPivot}
            />
          ))}

          {!results && !searching && !error && (
            <Text variant={TextVariant.BodyTertiary} color={TextColor.Muted}>
              Results appear here, one row per index. Every index lives in the same embedding space, so any query
              searches all of them: words find sounds, a sound finds videos, a video finds music.
            </Text>
          )}
        </Stack>
      </div>
    </div>
  );
}
