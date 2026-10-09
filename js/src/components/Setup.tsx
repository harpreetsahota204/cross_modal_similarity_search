import {
  Align,
  Button,
  EmptyState,
  IconName,
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

/** Shown until the dataset has an EmbeddingGemma 2 LanceDB index. */
export default function Setup({ uri, onBuild }: { uri?: string; onBuild: () => void }) {
  return (
    <Stack orientation={Orientation.Column} spacing={Spacing.Lg} align={Align.Center} style={{ padding: 32, maxWidth: 620, margin: "0 auto" }}>
      <EmptyState
        icon={IconName.Embeddings}
        title="No cross-modal indexes yet"
        description="EmbeddingGemma 2 puts text, images, video and audio in one embedding space. Build an index of this dataset's media, then search it with words, with other samples, or with a file."
      />
      <Button size={Size.Md} variant={Variant.Primary} leadingIcon={IconName.Add} onClick={onBuild}>
        Build an index
      </Button>
      <Text variant={TextVariant.Caption} color={TextColor.Muted} style={{ textAlign: "center" }}>
        Indexes are FiftyOne similarity indexes with the LanceDB backend, stored in {uri ?? "the brain's LanceDB URI"}.
        Indexes you create in Python with model="google/embeddinggemma-2" and backend="lancedb" appear here too.
      </Text>
    </Stack>
  );
}
