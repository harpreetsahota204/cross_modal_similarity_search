import {
  Align,
  Button,
  Icon,
  IconName,
  Justify,
  Orientation,
  RichList,
  Size,
  Spacing,
  Stack,
  Text,
  TextColor,
  TextVariant,
  Tooltip,
  Variant,
} from "@voxel51/voodo";
import React, { useMemo } from "react";
import { MODALITY_LABELS } from "../format";
import { PAGE, TOOLTIP_TEXT } from "../styles";
import type { IndexInfo } from "../types";

type Props = {
  indexes: IndexInfo[];
  uri: string | null;
  onBack: () => void;
  onBuild: () => void;
  onRebuild: (key: string) => void;
};

/** The dataset's EmbeddingGemma 2 LanceDB indexes. */
export default function Indexes({ indexes, uri, onBack, onBuild, onRebuild }: Props) {
  const listItems = useMemo(
    () =>
      indexes.map((i) => ({
        id: i.key,
        data: {
          primaryContent: (
            <Stack orientation={Orientation.Column} spacing={Spacing.Xs}>
              <span style={{ fontWeight: "bold" }}>{i.key}</span>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Embeds: {MODALITY_LABELS[i.modality]}
              </Text>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Model: google/embeddinggemma-2 ({i.dim}d)
              </Text>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Backend: lancedb
              </Text>
              {i.table && (
                <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                  Table: {i.table}
                </Text>
              )}
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Embeddings field: {i.field}
              </Text>
              <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                Vectors: {i.count}
              </Text>
              {i.slices && (
                <Text variant={TextVariant.BodySecondary} color={TextColor.Muted}>
                  Slices: {i.slices.join(", ")}
                </Text>
              )}
              <Stack orientation={Orientation.Row} spacing={Spacing.Xs} align={Align.Center}>
                <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                  Supports text queries?
                </Text>
                <Icon name={IconName.Check} size={Size.Sm} color={TextColor.Success} />
              </Stack>
              {!i.ok && (
                <Text variant={TextVariant.BodySecondary} color={TextColor.Failure}>
                  {i.error ?? "LanceDB table not found"}
                </Text>
              )}
            </Stack>
          ),
          actions: i.ok ? undefined : (
            <Tooltip content={<span style={TOOLTIP_TEXT}>Recreate the table from {i.field}</span>}>
              <Button size={Size.Sm} variant={Variant.Secondary} onClick={() => onRebuild(i.key)}>
                Rebuild table
              </Button>
            </Tooltip>
          ),
        },
      })),
    [indexes, onRebuild],
  );

  return (
    <Stack orientation={Orientation.Column} style={PAGE}>
      <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center} style={{ marginBottom: "1rem" }}>
        <Button
          aria-label="Back to searches"
          size={Size.Md}
          variant={Variant.Borderless}
          leadingIcon={IconName.ArrowLeft}
          onClick={onBack}
        />
        <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
          Back to searches
        </Text>
      </Stack>

      <Stack
        orientation={Orientation.Row}
        align={Align.Center}
        justify={Justify.Between}
        style={{ marginBottom: "1rem" }}
      >
        <Text variant={TextVariant.BodySecondary} color={TextColor.Muted}>
          {uri ? `LanceDB at ${uri}` : ""}
        </Text>
        <Button variant={Variant.Primary} size={Size.Sm} leadingIcon={IconName.Add} onClick={onBuild}>
          Similarity Index
        </Button>
      </Stack>

      <RichList listItems={listItems} />
    </Stack>
  );
}
