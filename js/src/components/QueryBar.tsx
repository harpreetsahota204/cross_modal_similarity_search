import {
  Align,
  Button,
  Dropzone,
  IconName,
  Input,
  Orientation,
  Size,
  Spacing,
  Stack,
  Text,
  TextColor,
  TextVariant,
  ToggleSwitch,
  ToggleSwitchVariant,
  Variant,
} from "@voxel51/voodo";
import React from "react";
import { MODALITY_LABELS, QUERY_MODALITIES, mediaKindOf } from "../format";
import type { Modality, QueryMode, Selection, UploadedQuery } from "../types";

const MODES: QueryMode[] = ["text", "samples", "file"];
const MAX_UPLOAD_MB = 50;

type Props = {
  mode: QueryMode;
  onMode: (mode: QueryMode) => void;
  text: string;
  onText: (text: string) => void;
  selection: Selection | null;
  upload: UploadedQuery | null;
  onUpload: (upload: UploadedQuery | null) => void;
  modality: Modality | null;
  onModality: (modality: Modality) => void;
  canSearch: boolean;
  searching: boolean;
  onSearch: () => void;
  onError: (message: string) => void;
};

/** The three ways to ask: text, the grid's selection, or a file. */
export default function QueryBar(props: Props) {
  const { mode, onMode } = props;

  const tabs = [
    { id: "text", data: { label: "Text", content: <TextQuery {...props} /> } },
    {
      id: "samples",
      data: {
        label: props.selection ? `Selection (${props.selection.count})` : "Selection",
        content: <SelectionQuery {...props} />,
      },
    },
    { id: "file", data: { label: "File", content: <FileQuery {...props} /> } },
  ];

  return (
    <ToggleSwitch
      variant={ToggleSwitchVariant.Soft}
      tabs={tabs}
      index={MODES.indexOf(mode)}
      onChange={(i) => onMode(MODES[i])}
    />
  );
}

function SearchButton({ canSearch, searching, onSearch }: Props) {
  return (
    <Button
      size={Size.Sm}
      variant={Variant.Primary}
      leadingIcon={IconName.Search}
      disabled={!canSearch || searching}
      onClick={onSearch}
    >
      {searching ? "Searching…" : "Search"}
    </Button>
  );
}

function TextQuery(props: Props) {
  const { text, onText, canSearch, onSearch } = props;
  return (
    <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center} style={{ paddingTop: 10 }}>
      <div style={{ flex: 1 }}>
        <Input
          size={Size.Sm}
          icon={IconName.Search}
          value={text}
          placeholder="Describe what to find: a dog barking, rain on a window, a red car at night…"
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => onText(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent) => {
            if (e.key === "Enter" && canSearch) onSearch();
          }}
        />
      </div>
      <SearchButton {...props} />
    </Stack>
  );
}

function ModalityChoice({
  options,
  value,
  onChange,
  note,
}: {
  options: Modality[];
  value: Modality | null;
  onChange: (m: Modality) => void;
  note?: (m: Modality) => string | null;
}) {
  return (
    <Stack orientation={Orientation.Row} spacing={Spacing.Xs} align={Align.Center} style={{ flexWrap: "wrap" }}>
      <Text variant={TextVariant.BodyTertiary} color={TextColor.Secondary}>
        Search with its
      </Text>
      {options.map((m) => (
        <Button
          key={m}
          size={Size.Xs}
          variant={m === value ? Variant.Primary : Variant.Secondary}
          onClick={() => onChange(m)}
        >
          {MODALITY_LABELS[m].toLowerCase()}
        </Button>
      ))}
      {value && note?.(value) && (
        <Text variant={TextVariant.Caption} color={TextColor.Muted}>
          {note(value)}
        </Text>
      )}
    </Stack>
  );
}

function SelectionQuery(props: Props) {
  const { selection, modality, onModality } = props;

  if (!selection) {
    return (
      <Hint>
        Select samples in the grid, then search every index with their image, video or audio. Several samples are
        averaged into one query.
      </Hint>
    );
  }

  if (selection.error) {
    return <Hint failure>{selection.error}</Hint>;
  }

  const noun = selection.count === 1 ? selection.kind : `${selection.kind}s`;
  return (
    <Stack orientation={Orientation.Row} spacing={Spacing.Md} align={Align.Center} style={{ paddingTop: 10, flexWrap: "wrap" }}>
      <Text variant={TextVariant.BodySecondary}>
        {selection.count} selected {noun}
      </Text>
      <div style={{ flex: 1 }}>
        <ModalityChoice
          options={selection.modalities}
          value={modality}
          onChange={onModality}
          note={(m) => (selection.stored[m] ? `vectors from ${selection.stored[m]}` : "embedded when you search")}
        />
      </div>
      <SearchButton {...props} />
    </Stack>
  );
}

function FileQuery(props: Props) {
  const { upload, onUpload, modality, onModality, onError } = props;

  const onFiles = (files: File[]) => {
    const file = files[0];
    if (!file) return;
    const kind = mediaKindOf(file.name);
    if (!kind) {
      onError(`Can't embed ${file.name}: use an image, video, audio or .mcap file`);
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      onError(`${file.name} is over ${MAX_UPLOAD_MB} MB`);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      onUpload({ name: file.name, data: String(reader.result), kind });
      onModality(QUERY_MODALITIES[kind][0]);
    };
    reader.readAsDataURL(file);
  };

  if (!upload) {
    return (
      <div style={{ paddingTop: 10 }}>
        <Dropzone
          title="Drop an image, video or audio file"
          description={`Searches every index with it. Up to ${MAX_UPLOAD_MB} MB; the file isn't added to the dataset`}
          accept="image/*,video/*,audio/*,.mcap"
          onFiles={onFiles}
        />
      </div>
    );
  }

  return (
    <Stack orientation={Orientation.Row} spacing={Spacing.Md} align={Align.Center} style={{ paddingTop: 10, flexWrap: "wrap" }}>
      <Text variant={TextVariant.BodySecondary}>{upload.name}</Text>
      <Button size={Size.Xs} variant={Variant.Borderless} leadingIcon={IconName.Close} onClick={() => onUpload(null)}>
        Remove
      </Button>
      <div style={{ flex: 1 }}>
        {upload.kind && (
          <ModalityChoice options={QUERY_MODALITIES[upload.kind]} value={modality} onChange={onModality} />
        )}
      </div>
      <SearchButton {...props} />
    </Stack>
  );
}

function Hint({ children, failure }: { children: React.ReactNode; failure?: boolean }) {
  return (
    <div style={{ paddingTop: 10 }}>
      <Text variant={TextVariant.BodyTertiary} color={failure ? TextColor.Failure : TextColor.Secondary}>
        {children}
      </Text>
    </div>
  );
}
