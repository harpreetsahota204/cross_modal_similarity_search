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

/** Forgets this panel's persistent state under a key prefix, so the next
 * mount starts from its initial values. */
export function useForgetPersistentState() {
  const panelId = usePanelId();
  return useCallback(
    (prefix: string) => {
      for (const id of Array.from(store.keys())) {
        if (id.startsWith(`${panelId}:${prefix}`)) store.delete(id);
      }
    },
    [panelId],
  );
}
