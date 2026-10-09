import {
  Align,
  Button,
  FormField,
  Heading,
  HeadingLevel,
  IconName,
  Input,
  InputType,
  Justify,
  Orientation,
  RadioGroup,
  Select,
  Size,
  Spacing,
  Spinner,
  Stack,
  Text,
  TextArea,
  TextColor,
  TextVariant,
  Variant,
} from "@voxel51/voodo";
import React, { useState } from "react";
import { MIDDLE_DOT, MODALITY_LABELS, QUERY_MODALITIES } from "../format";
import { usePersistentState } from "../hooks/usePersistentState";
import { ACTIVE_BOX, INFO_CARD, PAGE } from "../styles";
import type { FormSeed, IndexInfo, Modality, QueryMode, Selection, UploadedQuery } from "../types";
import MediaInput from "./MediaInput";

const MAX_K = 100;

// Which part of the query media becomes the query vector
const MODALITY_HELP: Record<Modality, string> = {
  image: "The picture becomes the query vector",
  video: "Only the frames become the query vector; the sound is ignored",
  audio: "Only the sound becomes the query vector; the picture is ignored",
  "video+audio": "Frames and sound together become one query vector",
};

const QUERY_TYPES: { mode: QueryMode; label: string; icon: IconName }[] = [
  { mode: "text", label: "Text", icon: IconName.Search },
  { mode: "samples", label: "Samples", icon: IconName.ImageSearch },
  { mode: "file", label: "Media", icon: IconName.Upload },
];

type Props = {
  indexes: IndexInfo[];
  selection: Selection | null;
  seed: FormSeed | null;
  modelLoaded: boolean;
  onBack: () => void;
  onSearch: (params: Record<string, unknown>) => Promise<void>;
};

export default function NewSearch({ indexes, selection, seed, modelLoaded, onBack, onSearch }: Props) {
  const usable = indexes.filter((i) => i.ok);
  const [storedMode, setMode] = usePersistentState<QueryMode>("form.mode", seed?.mode ?? "text");
  const mode = QUERY_TYPES.some((t) => t.mode === storedMode) ? storedMode : "file";
  const [text, setText] = usePersistentState("form.text", seed?.text ?? "");
  const [keys, setKeys] = usePersistentState<string[] | null>("form.keys", seed?.brainKeys ?? null);
  const [k, setK] = usePersistentState<number | "">("form.k", seed?.k ?? 12);
  const [modality, setModality] = usePersistentState<Modality | null>("form.modality", seed?.modality ?? null);
  const [upload, setUpload] = usePersistentState<UploadedQuery | null>("form.upload", null);
  const [name, setName] = usePersistentState("form.name", "");
  const [submitting, setSubmitting] = useState(false);

  const brainKeys = (keys ?? usable.map((i) => i.key)).filter((key) => usable.some((i) => i.key === key));
  const chosen = usable.filter((i) => brainKeys.includes(i.key));

  const media = mode === "file" ? upload : null;

  // The modality radio offers what the query's media can be embedded as
  const options: Modality[] =
    mode === "samples"
      ? selection?.modalities ?? []
      : media?.kind
        ? QUERY_MODALITIES[media.kind]
        : [];
  const preferred = mode === "samples" ? options.find((m) => selection?.stored[m]) : undefined;
  const activeModality = modality && options.includes(modality) ? modality : preferred ?? options[0] ?? null;

  const kValid = typeof k === "number" && k >= 1 && k <= MAX_K;
  const hasQuery =
    mode === "text"
      ? text.trim().length > 0
      : mode === "samples"
        ? !!selection && !selection.error && !!activeModality
        : !!media && !!activeModality;
  const canSubmit = chosen.length > 0 && kValid && hasQuery && !submitting;

  const submit = async () => {
    if (!canSubmit) return;
    const query =
      mode === "text"
        ? { mode, text }
        : mode === "samples"
          ? { mode, sample_ids: selection?.ids ?? [], modality: activeModality }
          : { mode, file: media, modality: activeModality };
    setSubmitting(true);
    try {
      await onSearch({ ...query, brain_keys: brainKeys, k, run_name: name.trim() });
      setName("");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={PAGE}>
      <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center} style={{ marginBottom: "1.5rem" }}>
        <Button size={Size.Md} variant={Variant.Borderless} leadingIcon={IconName.ArrowLeft} onClick={onBack} />
        <Heading level={HeadingLevel.H2}>{seed ? "Clone Search" : "New Search"}</Heading>
      </Stack>

      <Stack orientation={Orientation.Column} spacing={Spacing.Lg}>
        <FormField
          label="Search against"
          description="The indexes your query is compared to. Pick more than one to get a separate set of matches from each"
          control={
            <Select
              exclusive={false}
              value={brainKeys}
              onChange={(v) => setKeys(Array.isArray(v) ? v : v ? [v] : [])}
              options={usable.map((i) => ({
                id: i.key,
                data: { label: `${i.key}: samples embedded as ${MODALITY_LABELS[i.modality].toLowerCase()}` },
              }))}
            />
          }
        />

        {chosen.length > 0 && (
          <div style={INFO_CARD}>
            <Stack orientation={Orientation.Column} spacing={Spacing.Xs}>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Model: google/embeddinggemma-2 {MIDDLE_DOT} Backend: lancedb {MIDDLE_DOT} Metric: cosine
              </Text>
              {chosen.map((i) => (
                <Text key={i.key} variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                  {i.key}: {MODALITY_LABELS[i.modality]} {MIDDLE_DOT} {i.count} vectors {MIDDLE_DOT} {i.dim}d
                  {i.slices ? ` ${MIDDLE_DOT} slices ${i.slices.join(", ")}` : ""}
                </Text>
              ))}
            </Stack>
          </div>
        )}

        <Stack orientation={Orientation.Row} spacing={Spacing.Xs} style={{ flexWrap: "wrap" }}>
          {QUERY_TYPES.map((t) => (
            <Button
              key={t.mode}
              size={Size.Sm}
              variant={mode === t.mode ? Variant.Primary : Variant.Secondary}
              leadingIcon={t.icon}
              onClick={() => setMode(t.mode)}
              style={{ flex: "1 1 90px", whiteSpace: "nowrap" }}
            >
              {t.label}
            </Button>
          ))}
        </Stack>

        {mode === "text" && (
          <FormField
            label="Text query"
            control={
              <TextArea
                rows={3}
                size={Size.Sm}
                value={text}
                placeholder="Describe what to find: a dog barking, rain on a window, a red car at night…"
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
                onKeyDown={(e: React.KeyboardEvent) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
            }
          />
        )}

        {mode === "samples" &&
          (selection && !selection.error ? (
            <div style={ACTIVE_BOX}>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Primary}>
                {selection.count} {selection.count === 1 ? selection.kind : `${selection.kind}s`} selected
                {selection.count > 1 ? "; their vectors are averaged into one query" : ""}
              </Text>
            </div>
          ) : (
            <div style={INFO_CARD}>
              <Text
                variant={TextVariant.BodySecondary}
                color={selection?.error ? TextColor.Failure : TextColor.Secondary}
              >
                {selection?.error ?? "Select samples in the grid"}
              </Text>
            </div>
          ))}

        {mode === "file" && (
          <MediaInput
            value={upload}
            onChange={(v) => {
              setUpload(v);
              if (v?.kind) setModality(QUERY_MODALITIES[v.kind][0]);
            }}
          />
        )}

        {options.length > 0 && (
          <FormField
            label="Embed your query as"
            description={
              activeModality &&
              [
                MODALITY_HELP[activeModality],
                mode === "samples"
                  ? selection?.stored[activeModality]
                    ? `Uses the vectors already stored in ${selection.stored[activeModality]}`
                    : "Embedded when you search"
                  : null,
              ]
                .filter(Boolean)
                .join(". ")
            }
            control={
              <RadioGroup
                options={options.map((m) => ({ value: m, label: MODALITY_LABELS[m] }))}
                value={activeModality ?? undefined}
                onChange={(v) => setModality(v as Modality)}
                style={{ display: "flex", flexDirection: "row", gap: "1rem" }}
              />
            }
          />
        )}

        <FormField
          label="Number of matches"
          description="Per index"
          error={kValid ? undefined : `Enter 1 to ${MAX_K}`}
          control={
            <Input
              type={InputType.Number}
              size={Size.Sm}
              min={1}
              max={MAX_K}
              value={k === "" ? "" : String(k)}
              error={!kValid}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setK(e.target.value === "" ? "" : parseInt(e.target.value, 10))
              }
            />
          }
        />

        <FormField
          label="Search name (optional)"
          control={
            <Input
              size={Size.Sm}
              placeholder="Defaults to the query"
              value={name}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
            />
          }
        />

        <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center} justify={Justify.End}>
          {submitting && (
            <>
              <Spinner size={Size.Sm} />
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                {modelLoaded ? "Searching…" : "Loading EmbeddingGemma 2 (about 15 s, first search only)"}
              </Text>
            </>
          )}
          <Button variant={Variant.Primary} size={Size.Sm} disabled={!canSubmit} onClick={submit}>
            {submitting ? "Searching..." : "Search"}
          </Button>
        </Stack>
      </Stack>
    </div>
  );
}
