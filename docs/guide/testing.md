# Testing

Two test levels, both optional for everyday annotation and both valuable after any change to the backend or config.

## Smoke test — `make test`

```bash
make test   # needs: make up + a running backend
```

Runs `scripts/smoke_test.py`, which exercises the same HTTP calls the UI makes:

1. Label Studio health + legacy API token check.
2. Creates a throwaway `smoke-test-*` project, uploads the COCO "two cats" photo (640×480).
3. Connects the backend and calls `/setup` the way Label Studio does.
4. Sends click and box prompts **directly** to the backend (`/predict`) and **through Label Studio** (`/api/ml/{id}/interactive-annotating`).
5. Rasterises the returned masks and asserts: correct RLE brush type/label, click point inside the mask, and **bbox IoU > 0.75** vs. ground-truth cat boxes.
6. Tests positive + negative click refinement, and measures latency (cold = embedding computed, warm = cached; cold/warm medians on never-seen images).
7. Creates annotations from predictions; checks the native `BRUSH_TO_PNG` export; runs `export_coco.py`, `export_yolo.py`, `export_masks.py` and validates their output (2 COCO annotations, YOLO class-0 lines, instance IDs {1,2}, semantic classes {0,1}, polygon area within 3% of brush area).

**Outputs:** overlays (`overlay_*.png`), exported files, and `smoke_results.json` in `test_outputs/`. Exit code is non-zero with a `FAILED:` list if any check fails; `ALL CHECKS PASSED` otherwise.

## UI test — `make ui-test`

Drives the **real Label Studio UI** in headless Chromium via Playwright (`scripts/ui_test.py`): login → open task → Auto-Annotation ON → click prompt → negative-click refinement → accept → box prompt → accept → Submit → Update → exports, checking the backend log and stored annotation along the way.

One-time browser install:

```bash
(cd sam3_backend && uv run playwright install chromium)
```

```bash
make ui-test                 # default project 6
make ui-test UI_PROJECT=3    # choose another project
```

**Outputs:** screenshots `test_outputs/ui_*.png` (task open, suggestion, accepted masks, submitted, updated…).

::: danger Destructive to one task
`make ui-test` **deletes the annotations and drafts of the first task** of the target project (`UI_PROJECT=<id>` to pick another). Drafts are deleted deliberately — Label Studio would otherwise restore old regions. Do not point it at a project with annotations you care about.
:::

## What "passing" tells you

| Check | Proves |
|---|---|
| Direct `/predict` click/box IoU > 0.75 | SAM3 weights, MPS, prompt-coordinate conversion all work |
| Via-LS interactive-annotating result | Container → `host.docker.internal:9090` networking + interactive mode work |
| Exporter checks | RLE decode → polygon/box/mask conversion is correct end-to-end |
| UI test green | The real browser flow (toggle, magic tool, suggestions, submit) works |
