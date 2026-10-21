# Exporting Annotations

Set `PROJECT` to your project ID (from the URL — `make setup` prints it) and `OUT` to the output directory (default `exports/`).

## Commands

| Format | Command | Output |
|---|---|---|
| COCO (polygons + bboxes; brush masks converted) | `make export-coco PROJECT=1` | `exports/coco/result.json` (category IDs 1-based) |
| YOLO detection + segmentation | `make export-yolo PROJECT=1` | `exports/yolo/yolo_box/` (`cls cx cy w h`) and `yolo_seg/` (`cls x1 y1 ...`), each with `images/`, `labels/`, `classes.txt`, `data.yaml` |
| PNG masks | `make export-masks PROJECT=1` | `exports/masks/<img>_semantic.png` (0 = bg, class id + 1), `<img>_instance.png` (16-bit, 1..N), `<img>_instance.json`, `classes.json` |
| Raw Label Studio JSON | `make export-json PROJECT=1` | `exports/project.json` |
| All of the above | `make export-all PROJECT=1` | |

The scripts read annotations through the LS API. Boxes are derived from the mask/polygon extents; brush RLE is decoded and converted to polygons for COCO and YOLO-seg.

## Details per format

### COCO — `scripts/export_coco.py`

Instance segmentation with polygon segmentations and bounding boxes. Category IDs are 1-based and follow the `label_config.xml` order. Also accepts an offline export file:

```bash
cd sam3_backend && uv run python ../scripts/export_coco.py --export ../exports/project.json --out ../exports/coco/result.json
```

### YOLO — `scripts/export_yolo.py`

Produces **both** flavours in one run:

- `yolo_box/labels/*.txt` — detection: `cls cx cy w h` (box = mask/polygon bounding box)
- `yolo_seg/labels/*.txt` — segmentation: `cls x1 y1 x2 y2 ...` (normalised polygon, one line per contour)

Images are copied when found (`--images <dir>` or downloaded from Label Studio). Each flavour gets `classes.txt` and a ready-to-train `data.yaml`.

### PNG masks — `scripts/export_masks.py`

- **Semantic** (`<stem>_semantic.png`, 8-bit): 0 = background, pixel value = class id + 1. Overlapping objects: later wins.
- **Instance** (`<stem>_instance.png`, 16-bit): 0 = background, 1..N per object in annotation order, plus `<stem>_instance.json` mapping instance id → label.
- `classes.json` legend is always written. `--mode semantic|instance|both` (default `both`).

No image files are needed — dimensions come from the annotations (`original_width/height`).

## Including hand-drawn rectangles

Pass `--rects` to `export_coco.py` / `export_yolo.py` / `export_masks.py` to also include hand-drawn `RectangleLabels` regions:

```bash
cd sam3_backend && uv run python ../scripts/export_yolo.py --project 1 --out ../exports/yolo --rects
```

By default rectangles are excluded — the box-prompt rectangles are SAM prompts and would duplicate their masks.

## What Label Studio's own exports do

::: danger Do not use Label Studio's built-in COCO/YOLO export for this project
Because the config contains `KeyPointLabels`, its COCO converter writes a single category `"default"` and its YOLO converter wrote **empty label files** (observed on 1.23.2). Use the `make export-*` scripts above.
:::

Label Studio's native **`BRUSH_TO_PNG`** and **`BRUSH_TO_NUMPY`** exports do work and are fine to use for masks.
