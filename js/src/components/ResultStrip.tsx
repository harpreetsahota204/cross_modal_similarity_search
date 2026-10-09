import {
  Align,
  Button,
  IconName,
  Justify,
  Orientation,
  Size,
  Spacing,
  Stack,
  Text,
  TextColor,
  TextVariant,
  Tooltip,
  Variant,
} from "@voxel51/voodo";
import React from "react";
import { BORDER, CARD, MODALITY_COLORS, basename, pivotModality, scoreFraction } from "../format";
import type { Hit, Modality, Section } from "../types";
import Media from "./Media";
import ModalityPill from "./ModalityPill";

const CARD_WIDTH = 196;
const MEDIA_HEIGHT = 128;

type Props = {
  section: Section;
  slices: string[] | null;
  onShowInGrid: (ids: string[]) => void;
  onOpen: (hit: Hit) => void;
  onPivot: (hit: Hit, modality: Modality) => void;
};

/** One index's results: a header and a horizontally scrolling row of cards. */
export default function ResultStrip({ section, slices, onShowInGrid, onOpen, onPivot }: Props) {
  const ids = section.hits.map((h) => h.id);

  return (
    <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
      <Stack orientation={Orientation.Row} align={Align.Center} justify={Justify.Between}>
        <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Sm}>
          <ModalityPill modality={section.modality} />
          <Text variant={TextVariant.HeadingSm}>{section.key}</Text>
          {slices && slices.length > 0 && (
            <Text variant={TextVariant.Caption} color={TextColor.Secondary}>
              slice {slices.join(", ")}
            </Text>
          )}
        </Stack>
        <Button
          size={Size.Xs}
          variant={Variant.Secondary}
          leadingIcon={IconName.GridView}
          disabled={ids.length === 0}
          onClick={() => onShowInGrid(ids)}
        >
          Show in grid
        </Button>
      </Stack>

      {section.error ? (
        <Text variant={TextVariant.BodyTertiary} color={TextColor.Failure}>
          {section.error}
        </Text>
      ) : section.hits.length === 0 ? (
        <Text variant={TextVariant.BodyTertiary} color={TextColor.Secondary}>
          No results
        </Text>
      ) : (
        <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 6 }}>
          {section.hits.map((hit, i) => (
            <ResultCard
              key={hit.id}
              rank={i + 1}
              hit={hit}
              modality={section.modality}
              onOpen={() => onOpen(hit)}
              onPivot={onPivot}
            />
          ))}
        </div>
      )}
    </Stack>
  );
}

function ResultCard({
  rank,
  hit,
  modality,
  onOpen,
  onPivot,
}: {
  rank: number;
  hit: Hit;
  modality: Modality;
  onOpen: () => void;
  onPivot: (hit: Hit, modality: Modality) => void;
}) {
  const pivot = pivotModality(hit.media, modality);
  const name = basename(hit.filepath);

  return (
    <div
      style={{
        flex: `0 0 ${CARD_WIDTH}px`,
        border: `1px solid ${BORDER}`,
        borderRadius: 8,
        background: CARD,
        padding: 8,
        boxSizing: "border-box",
      }}
    >
      <Stack orientation={Orientation.Column} spacing={Spacing.Xs}>
        <Media
          filepath={hit.filepath}
          kind={hit.media}
          muted={modality === "video" || modality === "image"}
          width={CARD_WIDTH - 18}
          height={MEDIA_HEIGHT}
        />

        <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Sm}>
          <Text variant={TextVariant.Caption} color={TextColor.Secondary}>
            #{rank}
          </Text>
          <div style={{ flex: 1, height: 4, borderRadius: 2, background: BORDER }}>
            <div
              style={{
                width: `${100 * scoreFraction(hit.score)}%`,
                height: "100%",
                borderRadius: 2,
                background: MODALITY_COLORS[modality],
              }}
            />
          </div>
          <Text variant={TextVariant.CodeSecondary}>{hit.score.toFixed(3)}</Text>
        </Stack>

        <Tooltip content={hit.label ? `${hit.label}\n${name}` : name}>
          <div style={{ minWidth: 0 }}>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              <Text variant={TextVariant.BodyTertiary}>{hit.label ?? name}</Text>
            </div>
            {hit.label && (
              <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                <Text variant={TextVariant.Caption} color={TextColor.Muted}>
                  {name}
                </Text>
              </div>
            )}
          </div>
        </Tooltip>

        <Stack orientation={Orientation.Row} spacing={Spacing.Xs}>
          <Button size={Size.Xs} variant={Variant.Secondary} leadingIcon={IconName.ExternalLink} onClick={onOpen}>
            Open
          </Button>
          {pivot && (
            <Tooltip content={`Search every index with this sample's ${pivot}`}>
              <Button
                size={Size.Xs}
                variant={Variant.Secondary}
                leadingIcon={IconName.ImageSearch}
                onClick={() => onPivot(hit, pivot)}
              >
                Similar
              </Button>
            </Tooltip>
          )}
        </Stack>
      </Stack>
    </div>
  );
}
