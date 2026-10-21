# Scripts Reference

All scripts run via `uv` from `sam3_backend/` (e.g. `cd sam3_backend && uv run python ../scripts/<script>.py`), which is what the Makefile targets do. Shared behaviour (`.env` loading, API session, label parsing, RLE/polygon geometry) lives in `scripts/lsutil.py`.

## `setup_project.py`

Create or reuse the project, connect the backend, import images (`make setup`).

```bash
uv run python ../scripts/setup_project.py [--title "SAM3 annotation"] [--images ../images] \
  [--backend-url http://host.docker.internal:9090] [--label-config ../label_config.xml] [--update-config]
```

- Waits up to ~2 min for `GET /health`; exits with "run `make up` first" otherwise.
- Reuses a project with the same `--title`; with `--update-config`, PATCHes `label_config.xml` onto it (and explains the removed-label refusal if LS rejects it).
- Connects the ML backend (`is_interactive: true`) if that URL isn't already connected; a 401 gets a specific legacy-token hint.
- Imports `.jpg/.jpeg/.png/.webp/.bmp` from `--images` and prints the project URL.

## `backend_ctl.sh`

`start | stop | restart | status` for the background backend (`logs/backend.pid`, `logs/backend.log`). Stop also kills any listener on :9090 if the wrapper PID is gone.

## `export_coco.py`

COCO instance segmentation (polygons + bboxes, 1-based category IDs from `label_config.xml` order). Brush RLE is converted to polygons. Exists because LS's built-in COCO export collapses categories when `KeyPointLabels` is present.

```bash
uv run python ../scripts/export_coco.py --project 1 --out ../exports/coco/result.json
uv run python ../scripts/export_coco.py --export ../exports/project.json --out ../exports/coco/result.json [--rects]
```

## `export_yolo.py`

Both YOLO flavours (`yolo_box/`: `cls cx cy w h`; `yolo_seg/`: `cls x1 y1 ...`), each with `images/`, `labels/`, `classes.txt`, `data.yaml`. Images are copied from `--images` or downloaded from LS.

```bash
uv run python ../scripts/export_yolo.py --project 1 --out ../exports/yolo [--images ../images] [--rects]
```

## `export_masks.py`

Per-image PNG masks: `<stem>_semantic.png` (8-bit, class id + 1) and `<stem>_instance.png` (16-bit, 1..N) + `<stem>_instance.json` + `classes.json`.

```bash
uv run python ../scripts/export_masks.py --project 1 --out ../exports/masks [--mode semantic|instance|both] [--rects]
```

## `smoke_test.py` (`make test`)

End-to-end HTTP test: health/token checks, throwaway project, direct + via-LS click/box prompts with IoU > 0.75 assertions, latency measurement, annotation creation, and full exporter validation. Writes overlays + `smoke_results.json` to `test_outputs/`. See [Testing](/guide/testing).

## `ui_test.py` (`make ui-test`)

Playwright test driving the real UI (login → click/box prompts → submit/update → exports), asserting on the Regions panel, `interactive predict:` backend-log lines, and stored annotations. Screenshots to `test_outputs/ui_*.png`.

```bash
uv run python ../scripts/ui_test.py [--project 6] [--headed]
```

::: warning Destructive
It deletes the existing annotations **and drafts** of the first task of the target project. Drafts must go — LS would otherwise restore old regions.
:::

## `lsutil.py` (shared helpers)

| Helper | Purpose |
|---|---|
| `load_env()` / `ls_session()` | Minimal `.env` loader (no override of set vars); `Token`-auth `requests.Session` |
| `labels_from_config()` | Ordered labels from `BrushLabels`/`PolygonLabels` (index = class id) |
| `iter_regions()` | Yield `(region, label)` for brush/polygon (latest non-cancelled annotation) |
| `region_mask()` | Region → uint8 mask (RLE decode via `decode_rle`, polygon fill, or rectangle) |
| `mask_polygons()` / `region_geometry()` | Contours → simplified polygons, bbox, area |
| `image_stem()` | Strip LS's 8-hex upload prefix from filenames |
