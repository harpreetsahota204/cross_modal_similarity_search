import {
  Align,
  Button,
  EmptyState,
  IconName,
  Justify,
  Orientation,
  Size,
  Spacing,
  Stack,
  Text,
  TextColor,
  TextVariant,
  Variant,
} from "@voxel51/voodo";
import React from "react";
import { INFO_CARD, PAGE } from "../styles";

/** Shown until the dataset has an EmbeddingGemma 2 LanceDB index. */
export default function Setup({ uri, onBuild }: { uri?: string; onBuild: () => void }) {
  return (
    <div style={PAGE}>
      <Stack
        orientation={Orientation.Column}
        align={Align.Center}
        justify={Justify.Center}
        spacing={Spacing.Lg}
        style={{ ...INFO_CARD, height: "100%", minHeight: 320, boxSizing: "border-box" }}
      >
        <EmptyState
          icon={IconName.Embeddings}
          title="Cross-modal indexes let you search images, video and audio with words, samples or files"
          description="EmbeddingGemma 2 puts every modality in one embedding space. Build an index of this dataset's media to start searching."
        />
        <Button size={Size.Sm} variant={Variant.Primary} onClick={onBuild}>
          Build Cross-Modal Index
        </Button>
        <Text variant={TextVariant.BodySecondary} color={TextColor.Muted} style={{ textAlign: "center", maxWidth: 520 }}>
          Indexes are similarity indexes with the LanceDB backend, stored in {uri ?? "the brain's LanceDB URI"}.
          Indexes created in Python with model="google/embeddinggemma-2" and backend="lancedb" appear here too.
        </Text>
      </Stack>
    </div>
  );
}
