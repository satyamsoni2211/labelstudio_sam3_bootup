# Overview

Label Studio + SAM3 gives you **click/box segmentation** inside Label Studio. You click on an object (or drag a box around it) and Meta's SAM3 returns an editable **brush mask** — no manual polygon tracing. The result is a normal annotation you can edit with the brush and eraser, relabel, delete, and export.

## Who this is for

- You annotate images on an **Apple Silicon Mac** and want model-assisted masks without a cloud GPU.
- You need training-ready exports: COCO, YOLO detection, YOLO segmentation, or PNG masks.
- You want everything local: images, annotations, and weights never leave your machine (except the one-time gated weight download).

## What runs where

| Piece | Runs as | Address | Purpose |
|---|---|---|---|
| Label Studio `1.23.2` | Docker container `label-studio` | `http://localhost:8080` | Annotation UI, projects, tasks, API |
| SAM3 ML backend | Native Python process on the Mac | `http://localhost:9090` | SAM3 inference on Apple GPU (MPS) |
| Export / setup scripts | `uv run python` on the Mac | — | Project setup, smoke tests, COCO/YOLO/mask exports |

The split matters. SAM3 needs the Apple GPU, which Docker on macOS cannot use — so the backend runs **natively**, and the container reaches it via `host.docker.internal:9090`. In the other direction, the backend downloads each image **through the Label Studio API** (token-authenticated), because uploaded files inside the container are not on the Mac's disk.

## The annotation loop

1. Open a task in the **SAM3 annotation** project.
2. Turn **Auto-Annotation ON** (bottom bar).
3. Pick the **magic Auto-Detect tool** (keypoint for clicks, rectangle for boxes), then a label.
4. Click or drag. SAM3's mask arrives as a **suggestion** in ~0.2 s (first prompt on an image ~1.5 s).
5. Refine with more clicks (**Alt/Option+click** = negative point), then accept with the green check.
6. Edit as a normal brush region if needed, then **Submit**.

The full step-by-step, including the two settings everyone misses, is in [Annotating with SAM3](/guide/annotating).

## Outputs

SAM3's mask becomes one of three region types depending on `SAM3_OUTPUT`:

| `SAM3_OUTPUT` | Result per prompt |
|---|---|
| `brush` (default) | One `BrushLabels` region (RLE-encoded mask) |
| `polygon` | One `PolygonLabels` region (largest connected component) |
| `both` | Both regions, overlapping — delete one |

Exports are handled by purpose-built scripts (not Label Studio's built-in COCO/YOLO converters — see [Exporting](/guide/exports) for why).

## Repository map

```
.
├── docker-compose.yml      # Label Studio container (:8080)
├── label_config.xml        # Brush / click / box / polygon label groups
├── Makefile                # up, backend, setup, test, export-* targets
├── .env.example            # HF_TOKEN, admin login, API token (copy to .env)
├── sam3_backend/
│   ├── sam3_model.py       # SAM3Backend: predict(), embedding cache, mask conversion
│   ├── _wsgi.py            # App entry: loads weights at startup, serves :9090
│   └── pyproject.toml      # torch, transformers, label-studio-ml (uv)
└── scripts/
    ├── setup_project.py    # Create/reuse project, connect backend, import images
    ├── backend_ctl.sh      # start|stop|restart|status for the background backend
    ├── smoke_test.py       # make test — end-to-end HTTP smoke test
    ├── ui_test.py          # make ui-test — drives the real UI (Playwright)
    ├── export_coco.py / export_yolo.py / export_masks.py
    └── lsutil.py           # Shared .env, API session, RLE/polygon helpers
```

## Next steps

- [Prerequisites](/guide/prerequisites) — Hugging Face gated access, `uv`, Docker. Do not skip the token step.
- [Quick Start](/guide/quick-start) — five commands to your first mask.
- [Installation in Detail](/guide/installation) — every step explained, with verification.
