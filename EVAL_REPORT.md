# Plugin Eval Report: @harpreetsahota/cross-modal-retrieval

## Summary
- **Security Gate:** PASS
- **Overall Score:** 86/100
- **Quality:** 88/100
- **Agent Readiness:** 78/100
- **Critical Issues:** 0
- **Warnings:** 3

## Security Assessment

No critical or suspicious findings. No `subprocess`, `eval`/`exec`, network
clients, `os.environ` access or `shutil`.

- `panel.py` `_file_query`: writes an uploaded query file into a
  `tempfile.TemporaryDirectory()` under a fixed name (`query` + extension),
  so the client's filename can't steer the path. The directory is removed
  after embedding. Uploads are capped at 50 MB on both sides.
- Model weights come from the Hugging Face Hub via the registered zoo source.

## Critical Issues

None.

## Warnings

1. **No `severity` on `build_index`** (`operators.py`, `BuildIndex.config`).
   It replaces an existing index with the same brain key. The form warns
   ("'<key>' exists and will be replaced"), but the App shows no risk badge.
   FiftyOne 1.22 has no `severity` option yet; add it when it does.
2. **LanceDB URI defaults to `/tmp/lancedb`** (`engine.lancedb_uri`). This is
   the backend's own default, and tables there vanish on reboot. Mitigated:
   the panel detects missing tables and rebuilds them from the stored field;
   the README shows how to set a permanent URI.
3. **Search operator output is a nested list** (`Search.resolve_output`), which
   the operator browser renders as raw JSON. The panel is the intended UI; the
   operator exists for SDK and agent use.

## Passed Checks

- Manifest: scoped name, semver, `fiftyone.version`, every operator and panel
  listed and registered; `js/dist/index.umd.js` built; `requirements.txt`
  present
- Registers cleanly (`build_plugin_contexts()` reports no errors)
- Indexes are native `fiftyone.brain` LanceDB similarity runs, discoverable by
  the App's similarity search and by indexes created in Python
- `build_index` uses `ctx.target_view()`, `execute_as_generator` with progress,
  and allows delegated execution
- Dynamic forms validate the brain key, count existing embeddings, and say
  how much work will run and on what device
- Model use is serialized under a lock (the App serves requests on threads)
- Store keys are dataset-scoped via `ctx.store()`, namespaced per panel, with
  a 24 h TTL
- Tests: 13 pytest (engine, upload path, build) and 5 vitest, all passing;
  verified in the App (text search, Similar pivot, Show in grid / Restore,
  build from the panel)

## Recommendations

1. Add a companion `SKILL.md` describing build-then-search for agents
2. Add `severity` to `build_index` once FiftyOne supports it
3. Consider an "Export results" action that saves a view per index

## Component Scores
| Area | Score (0–100) | Key Finding |
|------|--------------|-------------|
| Manifest & Structure | 95/100 | Complete; no `skills:` section |
| Security & Trust | 100/100 | No risky patterns; uploads sandboxed in temp dirs |
| Registration & MCP | 90/100 | Registers cleanly; both operators listed |
| Schema & Contract | 88/100 | Validated dynamic inputs; search output is raw JSON in the operator browser |
| Risk Classification | 75/100 | Replacement warned in-form, but no `severity` |
| Code Quality | 90/100 | Follows FiftyOne idioms; lint-clean |
| Agent Discoverability | 78/100 | Clear names and descriptions; no companion skill |
