# Label Studio + SAM3 click/box segmentation

Label Studio (Docker, :8080) with a SAM3 ML backend running **natively on the Mac** (Apple GPU via PyTorch MPS, :9090).
You click on an object (or drag a box) and SAM3 returns an editable **brush mask** (polygons optional, `SAM3_OUTPUT`). Exports: COCO, YOLO (boxes + segmentation), PNG masks.

```
browser --> Label Studio (docker, :8080) --http://host.docker.internal:9090--> SAM3 backend (native, MPS)
                                          <-- backend downloads images through the LS API (token in .env)
```

## Prerequisites
1. **Hugging Face access to `facebook/sam3`** (gated): accept the licence at https://huggingface.co/facebook/sam3 and
   use a token that can read gated repos (classic *Read* token, or a fine-grained token with
   "Read access to contents of all public gated repos you can access"). Put it in `.env` as `HF_TOKEN=...` (never commit it).
2. **uv** (https://docs.astral.sh/uv/). It installs the right Python (3.11/3.12) itself.
3. **Docker Desktop** (or compatible) with `docker compose`.
4. Apple Silicon Mac (falls back to CPU; override with `DEVICE=cpu|mps|cuda`). The first start downloads the weights (a few GB).

`cp .env.example .env` and fill in `HF_TOKEN`; the other values work as they are
(login `admin@example.com` / `admin12345`, API token `LABEL_STUDIO_API_KEY`, all in `.env`).

## Run it
```bash
make sync            # once: install backend deps (uv sync)
make up              # Label Studio on http://localhost:8080
make backend-start   # SAM3 backend in the background; log: logs/backend.log, pid: logs/backend.pid
make setup           # create project "SAM3 annotation", connect the backend (interactive), import ./images/*
make test            # optional: end-to-end smoke test (writes overlays and exports to test_outputs/)
(cd sam3_backend && uv run playwright install chromium)   # once, for the browser test
make ui-test         # optional: drives the real LS UI headlessly, screenshots in test_outputs/ui_*.png
```
Other targets: `make backend` (foreground), `backend-status`, `backend-stop`, `backend-logs`, `logs`, `down`.
Put your images in `./images/` and run `make setup` again (it reuses the project and uploads them), or use the Import button in the UI.

## How to annotate
Open http://localhost:8080 and the project "SAM3 annotation" (printed by `make setup`), then open a task.
1. **Turn Auto-Annotation ON**: toggle in the bottom bar. With it off, the click/box tools only draw plain
   keypoints/rectangles and never call SAM3. The toggle state is stored **per browser** (a new browser, profile
   or private window starts with it off).
2. Recommended once: gear icon in the bottom bar -> **Keep label selected after creating a region**. Without it the
   label is deselected after every click and the refinement clicks in step 5 do nothing.
3. Pick the **magic tool**: hover the "Auto-Detect" wand button at the bottom of the left-hand image toolbar (hotkey `M`)
   and choose the **keypoint** tool (click prompts) or the **rectangle** tool (box prompts). Choose the tool *first*
   and the label second (choosing the tool after the label deselects the label).
4. Pick a label in the **Click prompt** row (e.g. `cat`) or the **Box prompt** row.
5. **Click** the object (or **drag a box**). After ~0.2 s (about 1.5 s for the first prompt on an image) SAM3's mask
   shows up as a suggestion and the bottom bar says "1 suggestions". Refine it with more clicks; **Alt/Option+click**
   adds a negative point. Each click re-runs SAM3 with all points.
6. Accept the mask with the green check mark in the bottom bar (red cross = discard), or switch on
   **Auto-Accept Suggestions** to accept each result directly. It becomes a normal `brush` region with the label you picked.
7. Edit it with the **brush / eraser** tools (top of the toolbar), change its label, or delete it. The click/box
   prompt regions are only helpers; the exports ignore them.
8. **Submit** (or **Update** when editing an existing annotation).

Notes
- **"Test Request" in the ML backend settings returns `[]`. That is normal**: SAM3 only works on a click/box
  (interactive context), so a plain prediction request has nothing to segment. The backend logs
  `predict called without interactive context ...`. A connected backend with state "Connected" is what matters.
- The backend also gets annotation events (create/update/delete). SAM3 is not trained, so they are acknowledged and
  ignored (`LS event ... ignored` in the log).
- `make test` exercises the same HTTP calls the UI makes; `make ui-test` drives the real UI in headless Chromium
  (Playwright), including steps 1-8 above. **`make ui-test` deletes the annotations/drafts of the first task of project 6**
  (`UI_PROJECT=<id>` to use another project); screenshots go to `test_outputs/ui_*.png`.
- Output type: `SAM3_OUTPUT=brush` (default) | `polygon` | `both` (env var of the backend, e.g.
  `SAM3_OUTPUT=polygon scripts/backend_ctl.sh restart`). `both` returns two overlapping regions per prompt, so delete one of them.

## Edit labels
Edit `label_config.xml`: the same `<Label value="..."/>` list must appear in `brush`, `click_prompt`, `box_prompt` and `polygon`
(the mask takes the label of the prompt that produced it). Updating the config of a project with annotations can be
refused by Label Studio if they use labels you removed. Then run
`cd sam3_backend && uv run python ../scripts/setup_project.py --update-config`, or paste the XML in
Project Settings -> Labeling Interface. Class ids in YOLO/COCO/masks follow the label order in that file.

## Export
`PROJECT` = project id from the URL (`make setup` prints it), `OUT` = output dir (default `exports/`).

| Format | Command | Output |
|---|---|---|
| COCO (polygons + bboxes; brush masks are converted) | `make export-coco PROJECT=1` | `exports/coco/result.json` (category ids 1-based) |
| YOLO detection + segmentation | `make export-yolo PROJECT=1` | `exports/yolo/yolo_box/` (`cls cx cy w h`) and `yolo_seg/` (`cls x1 y1 ...`), each with images/, labels/, classes.txt, data.yaml |
| PNG masks | `make export-masks PROJECT=1` | `exports/masks/<img>_semantic.png` (0=bg, class id+1), `<img>_instance.png` (16-bit, 1..N), `_instance.json`, `classes.json` |
| Raw LS JSON | `make export-json PROJECT=1` | `exports/project.json` |
| All of the above | `make export-all PROJECT=1` | |

The scripts read the annotations through the LS API. Boxes are derived from the mask / polygon extents; brush RLE is decoded and converted to polygons for COCO and YOLO-seg. Label Studio's own `BRUSH_TO_PNG` / `BRUSH_TO_NUMPY` exports also work. Pass `--rects` to
`export_coco.py` / `export_yolo.py` / `export_masks.py` to also include hand-drawn rectangles.

**Do not use Label Studio's built-in COCO/YOLO export for this project.** Because the config contains KeyPointLabels,
its COCO converter writes a single category "default" and its YOLO converter wrote empty label files (observed on 1.23.2).

## Troubleshooting
- **Backend gets 401/403 downloading `facebook/sam3`**: the token lacks gated-repo read permission or the licence is not accepted (see Prerequisites). Details in `logs/backend.log`.
- **ML backend disconnected / "URL resolves to a reserved network address"**: `docker-compose.yml` sets `USE_DEFAULT_BANNED_SUBNETS=false` for this; recreate the container (`make up`) after editing. Test from the container: `docker exec label-studio curl -s http://host.docker.internal:9090/health`.
- **401 from the LS API / scripts**: legacy API tokens are only enabled when the organization is created. `docker-compose.yml` sets `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN` and the token, so a fresh `./ls-data` is fine. If `ls-data` was created without them: `make down && rm -rf ls-data && make up` (loses projects), or enable legacy tokens in Organization -> API Tokens Settings in the UI.
- **Nothing happens / only a red dot after a click**: Auto-Annotation must be ON **and** the magic keypoint tool selected (step 3), then the label (step 4). Check `make backend-logs` for `interactive predict:` lines; none means the browser never called the backend.
- **Only the first click works**: enable "Keep label selected after creating a region" (step 2) or re-pick the label before each click.
- **Old regions reappear in a task**: Label Studio restores autosaved drafts; delete the draft or reload after submitting.
- **Backend cannot download the image**: `LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` in `.env` must be valid from the Mac (`http://localhost:8080`).
- **Slow first click on an image**: the image embedding is computed once (~1.5 s on M4 Pro) and cached (6 images, `SAM3_EMBED_CACHE`); later clicks take ~10-30 ms. The first request after a start also warms up MPS (~2 s).
- **Unsupported MPS op errors**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is already set; force CPU with `DEVICE=cpu`.
- **Stale backend process / port in use**: `make backend-stop` (or `kill $(cat logs/backend.pid)`).
- Model dir: `MODEL_DIR` (default `sam3_backend/model_dir`) is set so annotation events no longer crash the SDK job manager.
- Polygon detail (`SAM3_OUTPUT=polygon|both`): `SAM3_POLY_EPSILON` (default 0.0015, smaller = more vertices) and `SAM3_POLY_MAX_POINTS` (400).
- The compose settings (legacy token, disabled SSRF checks) are for a local-only instance; do not expose port 8080 publicly with them.
