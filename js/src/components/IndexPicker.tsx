import {
  Align,
  Button,
  Checkbox,
  IconName,
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
import { BORDER } from "../format";
import type { IndexInfo } from "../types";
import ModalityPill from "./ModalityPill";

type Props = {
  indexes: IndexInfo[];
  enabled: string[];
  onEnabled: (keys: string[]) => void;
  onRebuild: (key: string) => void;
};

/** The indexes to search, one toggleable chip each. */
export default function IndexPicker({ indexes, enabled, onEnabled, onRebuild }: Props) {
  const toggle = (key: string, on: boolean) =>
    onEnabled(on ? [...enabled, key] : enabled.filter((k) => k !== key));

  return (
    <Stack orientation={Orientation.Row} spacing={Spacing.Sm} align={Align.Center} style={{ flexWrap: "wrap" }}>
      <Text variant={TextVariant.Label} color={TextColor.Secondary}>
        Search in
      </Text>
      {indexes.map((index) => (
        <Tooltip
          key={index.key}
          content={
            index.ok
              ? `${index.count} vectors (${index.dim}d) from field '${index.field}', LanceDB table '${index.table}'`
              : index.error ?? "Unavailable"
          }
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "3px 8px",
              border: `1px solid ${BORDER}`,
              borderRadius: 6,
              opacity: index.ok ? 1 : 0.8,
            }}
          >
            <Checkbox
              size={Size.Sm}
              checked={index.ok && enabled.includes(index.key)}
              disabled={!index.ok}
              onChange={(on) => toggle(index.key, on)}
            />
            <ModalityPill modality={index.modality} />
            <Text variant={TextVariant.BodyTertiary}>{index.key}</Text>
            <Text variant={TextVariant.Caption} color={TextColor.Muted}>
              {index.ok ? index.count : "table missing"}
            </Text>
            {!index.ok && (
              <Button
                size={Size.Xs}
                variant={Variant.Secondary}
                leadingIcon={IconName.Refresh}
                onClick={() => onRebuild(index.key)}
              >
                Rebuild
              </Button>
            )}
          </div>
        </Tooltip>
      ))}
    </Stack>
  );
}
