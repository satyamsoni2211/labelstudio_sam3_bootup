# Backend Internals

The backend is two files in `sam3_backend/`. This page is for developers modifying or debugging it.

## `_wsgi.py` — entry point

- Loads `.env` from the repo root, sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and `LABEL_STUDIO_ML_BACKEND_V2=true` (single shared model instance so `/setup` and `/predict` agree).
- Calls `load_model()` at startup — weights load once and **fail fast** on token/licence problems.
- Builds the app with `init_app(model_class=SAM3Backend, model_dir=MODEL_DIR)`.
- Replaces the SDK webhook with a **no-op**: Label Studio posts annotation events (`ANNOTATION_CREATED/UPDATED`, `START_TRAINING`, …); SAM3 is not trained, so they are acknowledged and ignored. (Forking a job holding MPS/torch state would be risky.)
- Serves on `0.0.0.0:$ML_BACKEND_PORT` (default 9090) when run directly.

## `sam3_model.py` — `SAM3Backend`

Built on `label_studio_ml` 1.0.9: `predict()` returns a plain list of prediction dicts; the parsed label config is available as `self.parsed_label_config`.

### Device selection

```python
def pick_device() -> str:
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
```

`MODEL_VERSION` is `sam3-tracker-<device>`.

### `predict(tasks, context)`

1. **No context → `[]`.** LS "Test Request" and background pre-annotation have no click/box, so there is nothing to segment. Logged as `predict called without interactive context ...` — expected behaviour, not an error.
2. **Output tags** are resolved from the parsed label config according to `SAM3_OUTPUT` (`brush` → `BrushLabels`, `polygon` → `PolygonLabels`). A missing tag raises `ValueError` naming the expected tag.
3. **Image loading** (`_load_image`): always downloads through the LS API via `get_local_path(..., image_dir="/nonexistent", hostname=LABEL_STUDIO_URL, access_token=LABEL_STUDIO_API_KEY)`. The `/nonexistent` dir deliberately disables the "look in the local LS media dir" shortcut — the container's upload dir is not on this Mac's disk. EXIF transposition is applied so coordinates match what LS displays.
4. **Prompt conversion**: context percentages → pixels. `keypointlabels` → `input_points`/`input_labels` (`is_positive` → 1/0); `rectanglelabels` → `input_boxes` (`xyxy`). The region label is taken from the prompt.
5. **Segmentation** (`_segment`): see caching below. A single click (one point, no box) uses `multimask_output=True` and keeps the best-IoU candidate; otherwise the single mask is used. MPS is synchronised after inference for accurate timing.
6. **Result building**: brush → RLE via `label_studio_converter.brush.mask2rle`; polygon → `mask_to_polygon()` (largest external contour, `approxPolyDP` with `SAM3_POLY_EPSILON`, capped at `SAM3_POLY_MAX_POINTS`, coordinates as % with 4 decimals). Logged as `interactive predict: N point(s), box=yes/no, label=... -> brushlabels, mask N px, score ...`.

### Embedding cache

`_embed_cache` is an LRU `OrderedDict` keyed by `path:mtime`, holding `(image_embeddings, [H, W])` for up to `SAM3_EMBED_CACHE` (6) images.

- **Miss**: preprocess the image and run the vision encoder once (~1.5 s on M4 Pro).
- **Hit**: skip preprocessing entirely; only prompt coordinates are normalised. Later clicks: ~10–30 ms.

### Training hooks

`fit()` returns `{}` and `process_event()` logs and ignores events — there is no training.

## Dependencies

`sam3_backend/pyproject.toml` (Python ≥3.11, <3.13): `torch 2.14.1`, `transformers 5.20.2` (`Sam3TrackerModel` / `Sam3TrackerProcessor`), `label-studio-ml 1.0.9`, `label-studio-converter`, `opencv-python-headless`, `numpy`, `pillow`, `gunicorn`, `rq`. Playwright is a dev-only dependency (for `ui_test.py`).
