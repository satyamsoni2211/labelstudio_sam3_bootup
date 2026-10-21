# Editing Labels

Labels live in **`label_config.xml`** at the repository root. The default set is `cat`, `dog`, `person`, `part`, `hole` — replace them with your own classes.

## The one rule

The same `<Label value="..."/>` list must appear in **all four** groups:

| Group | Tag | Role |
|---|---|---|
| Final mask | `BrushLabels` (`brush`) | The editable mask region |
| Click prompt | `KeyPointLabels` (`click_prompt`) | Click prompts (`smart="true"`) |
| Box prompt | `RectangleLabels` (`box_prompt`) | Box prompts (`smart="true"`) |
| Polygon | `PolygonLabels` (`polygon`) | Manual polygons; SAM3 fills it only with `SAM3_OUTPUT=polygon\|both` |

Why: **the mask takes the label of the prompt that produced it.** If a label exists in the prompt groups but not in `BrushLabels`, the returned mask has nowhere to land. Keep colours consistent too — each group's `<Label>` carries a `background` colour.

## Change the labels

1. Edit `label_config.xml` — add/remove/rename `<Label>` entries identically in all four groups.
2. Push the config to the existing project:

```bash
cd sam3_backend && uv run python ../scripts/setup_project.py --update-config
```

…or paste the XML manually in **Project Settings → Labeling Interface**.

::: warning Renaming labels with existing annotations
Updating the config of a project that already has annotations can be **refused** by Label Studio if those annotations use labels you removed. Delete or migrate those annotations first.
:::

## Class IDs follow the file order

Class IDs in **YOLO, COCO, and PNG-mask exports** follow the label order in `label_config.xml` (`labels_from_config()` reads the `BrushLabels` order; index = class id, COCO ids are 1-based). Reordering labels renumbers your classes — do it before you start annotating, not after.

## Choosing the output type

Set `SAM3_OUTPUT` in the **backend's** environment:

```bash
SAM3_OUTPUT=polygon scripts/backend_ctl.sh restart
```

| Value | Effect |
|---|---|
| `brush` (default) | One brush (RLE) region per prompt |
| `polygon` | One polygon region per prompt (largest connected component) |
| `both` | Two overlapping regions per prompt — delete one of them |

Polygon detail is tunable: `SAM3_POLY_EPSILON` (default `0.0015`; smaller = more vertices) and `SAM3_POLY_MAX_POINTS` (default `400`).
