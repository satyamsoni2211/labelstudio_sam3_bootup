# Quick Start

From clone to first SAM3 mask in five commands. Assumes [Prerequisites](/guide/prerequisites) are done — especially the Hugging Face gated access.

## 1. Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in **`HF_TOKEN`**. The other values work as they are:

```ini
HF_TOKEN=hf_...            # your gated-read token
LS_USER_EMAIL=admin@example.com
LS_PASSWORD=admin12345
LABEL_STUDIO_API_KEY=0123456789abcdef0123456789abcdef01234567
LABEL_STUDIO_URL=http://localhost:8080
ML_BACKEND_PORT=9090
SAM3_MODEL_ID=facebook/sam3
```

## 2. Install backend dependencies (once)

```bash
make sync        # cd sam3_backend && uv sync
```

## 3. Start Label Studio

```bash
make up          # http://localhost:8080
```

Wait until `docker compose logs` shows it healthy (the container has a `/health` healthcheck).

## 4. Start the SAM3 backend

```bash
make backend-start   # background; log: logs/backend.log, pid: logs/backend.pid
make backend-status  # "running pid ..." + /health response
```

First start downloads the weights (a few GB) and loads the model — allow ~1–3 minutes. Prefer foreground with live logs? `make backend`.

## 5. Create the project and import images

Put images in `./images/` first (or import later from the UI), then:

```bash
make setup
```

This creates (or reuses) the project **"SAM3 annotation"** with `label_config.xml`, connects the backend at `http://host.docker.internal:9090` in interactive mode, imports `./images/*`, and prints the project URL.

## 6. Annotate

Open the project URL (login `admin@example.com` / `admin12345`), open a task, and:

1. Toggle **Auto-Annotation ON** (bottom bar).
2. Gear icon → **Keep label selected after creating a region** (do this once).
3. Hover the **Auto-Detect** wand (hotkey `M`) → pick the **keypoint** tool (clicks) or **rectangle** tool (boxes).
4. Pick a label in the **Click prompt** (or **Box prompt**) row — e.g. `cat`.
5. **Click** the object (or drag a box). The mask appears as a suggestion; accept with the green check.
6. **Submit**.

Order matters: **tool first, label second** (choosing the tool after the label deselects it). Full details and refinement clicks: [Annotating with SAM3](/guide/annotating).

## Verify (optional but recommended)

```bash
make test        # end-to-end smoke test → overlays + exports in test_outputs/
make ui-test     # drives the real UI headlessly (Playwright) → test_outputs/ui_*.png
```

For `ui-test`, install the browser once: `(cd sam3_backend && uv run playwright install chromium)`.

## Stop / restart cheatsheet

```bash
make backend-stop     # stop the SAM3 backend
make backend-logs     # tail -f logs/backend.log
make down             # stop Label Studio (data stays in ./ls-data)
make logs             # follow Label Studio logs
```
