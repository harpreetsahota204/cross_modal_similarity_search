import {
  Align,
  Button,
  Heading,
  HeadingLevel,
  Icon,
  IconName,
  Justify,
  Orientation,
  Pill,
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
import { MIDDLE_DOT, MODALITY_LABELS, formatQuery, formatTime } from "../format";
import { HIGHLIGHT, PAGE, POINTER, TOOLTIP_TEXT, modalityChip } from "../styles";
import type { Applied, QueryMode, Run } from "../types";

const QUERY_ICONS: Record<QueryMode, IconName> = {
  text: IconName.Search,
  samples: IconName.ImageSearch,
  file: IconName.Upload,
};

const tip = (text: string) => <span style={TOOLTIP_TEXT}>{text}</span>;

/** Stops a click on an action from also applying the whole row. */
const stop =
  (fn: () => void): React.MouseEventHandler =>
  (e) => {
    e.stopPropagation();
    fn();
  };

type Props = {
  runs: Run[];
  hasIndexes: boolean;
  applied: Applied | null;
  onApply: (runId: string, key?: string) => void;
  onClone: (run: Run) => void;
  onDelete: (runId: string) => void;
  onRefresh: () => void;
  onNewSearch: () => void;
  onIndexes: () => void;
};

/** The saved searches, newest first; clicking one shows its results. */
export default function Home({
  runs,
  hasIndexes,
  applied,
  onApply,
  onClone,
  onDelete,
  onRefresh,
  onNewSearch,
  onIndexes,
}: Props) {
  const listItems = useMemo(
    () =>
      runs.map((run) => {
        const done = run.status === "completed" && run.result_count > 0;
        const isApplied = applied?.run_id === run.run_id;
        const found = run.sections.filter((s) => s.count > 0);
        const keys = run.sections.map((s) => s.key).join(", ");

        return {
          id: run.run_id,
          data: {
            onClick: done ? () => onApply(run.run_id) : undefined,
            style: { ...(done ? POINTER : {}), ...(isApplied ? HIGHLIGHT : {}) },
            primaryContent: (
              <Stack orientation={Orientation.Column} spacing={Spacing.Xs}>
                <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center}>
                  <Icon name={QUERY_ICONS[run.query_type]} size={Size.Lg} color={TextColor.Primary} />
                  <span
                    title={run.run_name}
                    style={{
                      fontWeight: "bold",
                      fontSize: "0.9rem",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      minWidth: 0,
                    }}
                  >
                    {run.run_name}
                  </span>
                  {run.status === "failed" ? (
                    <Pill size={Size.Sm} color={TextColor.Failure} isStatus>
                      Failed
                    </Pill>
                  ) : (
                    <Text variant={TextVariant.BodySecondary} color={TextColor.Muted}>
                      {run.result_count} {run.result_count === 1 ? "result" : "results"}
                    </Text>
                  )}
                </Stack>
                <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
                  {formatQuery(run)}
                  {keys ? ` ${MIDDLE_DOT} ${keys}` : ""} {MIDDLE_DOT} k={run.k}
                </Text>
                <Text variant={TextVariant.BodySecondary} color={TextColor.Muted}>
                  {formatTime(run.creation_time)}
                </Text>
                {run.status === "failed" && run.error && (
                  <Text variant={TextVariant.BodySecondary} color={TextColor.Failure}>
                    {run.error}
                  </Text>
                )}
              </Stack>
            ),
            actions: (
              <Stack orientation={Orientation.Row}>
                <Tooltip content={tip("Show results")}>
                  <Button
                    aria-label="Show results"
                    size={Size.Md}
                    variant={Variant.Borderless}
                    leadingIcon={IconName.GridView}
                    disabled={!done}
                    onClick={stop(() => onApply(run.run_id))}
                  />
                </Tooltip>
                <Tooltip
                  content={tip(run.query_type === "file" ? "File searches can't be cloned" : "Clone search")}
                >
                  <span>
                    <Button
                      aria-label="Clone search"
                      size={Size.Md}
                      variant={Variant.Borderless}
                      leadingIcon={IconName.ContentCopy}
                      disabled={run.query_type === "file"}
                      onClick={stop(() => onClone(run))}
                    />
                  </span>
                </Tooltip>
                <Tooltip content={tip("Delete")}>
                  <Button
                    aria-label="Delete"
                    size={Size.Md}
                    variant={Variant.Borderless}
                    leadingIcon={IconName.Delete}
                    onClick={stop(() => onDelete(run.run_id))}
                  />
                </Tooltip>
              </Stack>
            ),
            additionalContent:
              found.length > 1 ? (
                <Stack orientation={Orientation.Row} spacing={Spacing.Xs} style={{ flexWrap: "wrap" }}>
                  {found.map((s) => (
                    <Tooltip
                      key={s.key}
                      content={tip(
                        `Filter to the ${s.count} ${MODALITY_LABELS[s.modality].toLowerCase()} results from ${s.key}`,
                      )}
                    >
                      <Button
                        size={Size.Xs}
                        variant={Variant.Secondary}
                        style={modalityChip(s.modality, isApplied && applied?.key === s.key)}
                        onClick={stop(() => onApply(run.run_id, s.key))}
                      >
                        {MODALITY_LABELS[s.modality]} {MIDDLE_DOT} {s.key} ({s.count})
                      </Button>
                    </Tooltip>
                  ))}
                </Stack>
              ) : undefined,
          },
        };
      }),
    [runs, applied, onApply, onClone, onDelete],
  );

  return (
    <Stack orientation={Orientation.Column} style={PAGE}>
      <Stack
        orientation={Orientation.Row}
        align={Align.Center}
        justify={Justify.Between}
        style={{ marginBottom: "1rem" }}
      >
        <Heading level={HeadingLevel.H2}>
          {runs.length > 0
            ? `${runs.length} Cross-Modal ${runs.length === 1 ? "Search" : "Searches"}`
            : "Cross-Modal Search"}
        </Heading>
        <Stack orientation={Orientation.Row} spacing={Spacing.Sm}>
          <Tooltip content={tip("Refresh")}>
            <Button size={Size.Md} variant={Variant.Borderless} leadingIcon={IconName.Refresh} onClick={onRefresh} />
          </Tooltip>
          <Tooltip content={tip("Indexes")}>
            <Button size={Size.Md} variant={Variant.Borderless} leadingIcon={IconName.Settings} onClick={onIndexes} />
          </Tooltip>
          <Tooltip content={tip(hasIndexes ? "Start a new search" : "Build an index first")}>
            <span>
              <Button
                variant={Variant.Primary}
                size={Size.Sm}
                leadingIcon={IconName.Add}
                disabled={!hasIndexes}
                onClick={onNewSearch}
              >
                New Search
              </Button>
            </span>
          </Tooltip>
        </Stack>
      </Stack>

      {runs.length === 0 ? (
        <Stack
          orientation={Orientation.Column}
          align={Align.Center}
          justify={Justify.Center}
          spacing={Spacing.Sm}
          style={{ flex: 1 }}
        >
          <Text color={TextColor.Secondary}>No cross-modal searches yet</Text>
          <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
            Click "New Search" to find images, video or audio with text, selected samples or a file.
          </Text>
        </Stack>
      ) : (
        <RichList listItems={listItems} />
      )}
    </Stack>
  );
}
