# Architecture

Three actors, two HTTP hops, one MPS GPU.

```
┌─────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────┐
│ Browser │ ───► │ Label Studio (Docker :8080)  │ ───► │ SAM3 backend (native :9090) │
│         │ ◄─── │ heartexlabs/label-studio     │ ◄─── │ sam3_backend/_wsgi.py       │
└─────────┘      │ :1.23.2                      │      │ Sam3Tracker on MPS          │
                 └──────────────┬───────────────┘      └──────────────┬──────────────┘
                                │  backend downloads each image       │
                                └──────── through the LS API ─────────┘
                                     (LABEL_STUDIO_URL + API token)
```

## Why the backend is not in Docker

Docker Desktop on macOS has no access to the Apple GPU (Metal / MPS). Running SAM3 in a container would force CPU inference. So:

- **Label Studio** runs in Docker (`docker-compose.yml`, image `heartexlabs/label-studio:1.23.2`, port `8080`).
- **The SAM3 backend** runs natively via `uv` (`sam3_backend/`), where PyTorch can use `mps`. It falls back to CUDA, then CPU, or obeys the `DEVICE` override.

The container reaches the host through `host.docker.internal` (mapped via `extra_hosts: host-gateway`), and the backend is registered in Label Studio with the URL `http://host.docker.internal:9090`.

## Request flow: one click

1. With **Auto-Annotation ON**, clicking sends the interactive context (keypoints / rectangle drawn so far) to Label Studio, which POSTs it to the backend's `/predict`.
2. `SAM3Backend.predict()` (`sam3_backend/sam3_model.py`) reads the context. Without a context it returns `[]` — this is why **Test Request** in the ML backend settings "does nothing". That is expected.
3. The backend downloads the task image through the LS API (`get_local_path` with `LABEL_STUDIO_URL` + `LABEL_STUDIO_API_KEY`) and applies EXIF transposition, matching what Label Studio displays.
4. Click coordinates (% of width/height) become pixel `input_points` / `input_labels` (1 = positive, 0 = negative); a box becomes `input_boxes` in `xyxy` pixels.
5. SAM3 inference: the image embedding is computed **once per image** (~1.5 s on an M4 Pro) and cached (6 images, `SAM3_EMBED_CACHE`); later prompts on the same image take ~10–30 ms. A single click uses `multimask_output` and picks the best-IoU candidate.
6. The mask is returned as an RLE **brush** region (default), a simplified **polygon**, or both (`SAM3_OUTPUT`). The region carries the **label of the prompt** that produced it.
7. Label Studio shows it as a suggestion; accepting turns it into a normal editable region.

## What the backend deliberately ignores

Label Studio also POSTs annotation events (`ANNOTATION_CREATED`, `START_TRAINING`, …) to `/webhook`. SAM3 is not trained, so `_wsgi.py` replaces the webhook with a no-op that acknowledges and ignores them (`LS event ... ignored` in the log). `fit()` likewise returns `{}`.

## Startup behaviour

`_wsgi.py` calls `load_model()` at import time: weights load **once at startup**, failing fast if the Hugging Face token or gated access is wrong — rather than on your first click. `MODEL_DIR` (default `sam3_backend/model_dir`) gives the SDK job manager a writable directory so annotation events cannot crash it.

## Compose settings you should know about

`docker-compose.yml` is tuned for this local topology. Changing these breaks things in specific ways (see [Troubleshooting](/help/troubleshooting)):

| Setting | Why |
|---|---|
| `USE_DEFAULT_BANNED_SUBNETS=false` | `host.docker.internal` resolves to a "reserved" address Label Studio would otherwise refuse when registering the backend |
| `SSRF_PROTECTION_ENABLED=false` | Same local-only bypass |
| `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true` + `LABEL_STUDIO_USER_TOKEN` | Pre-seeds the legacy API token the backend and scripts authenticate with |
| `LOCAL_FILES_SERVING_ENABLED=true` | Serves `./images` mounted at `/label-studio/files/images` |

::: danger Local only
The SSRF and token settings above are for a **local-only** instance. Do not expose port 8080 publicly with them.
:::
