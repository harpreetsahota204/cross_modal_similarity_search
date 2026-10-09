import { useTriggerPanelEvent } from "@fiftyone/operators";
import { useCallback } from "react";
import type { PanelMethods } from "../types";

/** Calls a Python panel method by name, optionally awaiting completion. */
export function usePanelClient(methods: PanelMethods) {
  const trigger = useTriggerPanelEvent();

  return useCallback(
    (method: keyof PanelMethods, params: Record<string, unknown> = {}) =>
      new Promise<unknown>((resolve) => {
        trigger(methods[method], params, false, resolve);
      }),
    [trigger, methods],
  );
}
