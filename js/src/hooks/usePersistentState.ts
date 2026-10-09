import { usePanelId } from "@fiftyone/spaces";
import { useCallback, useState } from "react";

// The App re-mounts panel content on layout changes, and replaces its panel
// state objects when Python sends data, so UI state lives here, per panel
const store = new Map<string, unknown>();

export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, (next: T | ((prev: T) => T)) => void] {
  const id = `${usePanelId()}:${key}`;
  const [value, setValue] = useState<T>(() =>
    store.has(id) ? (store.get(id) as T) : initial,
  );

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved =
          typeof next === "function" ? (next as (prev: T) => T)(prev) : next;
        store.set(id, resolved);
        return resolved;
      });
    },
    [id],
  );

  return [value, set];
}
