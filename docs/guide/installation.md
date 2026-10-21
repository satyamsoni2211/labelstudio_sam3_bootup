# Installation in Detail

Every step of the setup, what it does, and how to verify it. If anything fails, jump to [Troubleshooting](/help/troubleshooting) — each failure mode there maps to a step here.

## Step 0 — Clone and enter

```bash
git clone git@github.com:satyamsoni2211/labelstudio_sam3_bootup.git
cd labelstudio_sam3_bootup
```

## Step 1 — Hugging Face gated access

1. Accept the licence at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) while logged in.
2. Create a token: classic **Read**, or fine-grained with **"Read access to contents of all public gated repos you can access"**.

**Verify:** `curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/models/facebook/sam3` should return model JSON, not a 401/403/gated error. (Export `HF_TOKEN` in your shell first, or check after Step 2.)

## Step 2 — Create `.env`

```bash
cp .env.example .env
```

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Hugging Face token from Step 1. Never commit. |
| `LS_USER_EMAIL` / `LS_PASSWORD` | Admin account created on the container's first start |
| `LABEL_STUDIO_API_KEY` | Legacy API token (40 hex chars). Used by the backend to download images and by all `scripts/*.py` |
| `LABEL_STUDIO_URL` | Where the **host** reaches Label Studio: `http://localhost:8080` |
| `ML_BACKEND_PORT` | Backend port: `9090` |
| `SAM3_MODEL_ID` | `facebook/sam3` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (already set; lets unsupported MPS ops fall back to CPU) |

The token is pre-seeded into the container via `LABEL_STUDIO_USER_TOKEN`, so a fresh `./ls-data` works out of the box. See the 401 entry in [Troubleshooting](/help/troubleshooting) if yours predates this.

## Step 3 — Install the backend environment

```bash
make sync    # cd sam3_backend && uv sync
```

`uv` creates `sam3_backend/.venv` with Python 3.11/3.12 and installs `torch`, `transformers`, `label-studio-ml`, `opencv-python-headless`, and friends (`sam3_backend/pyproject.toml`). This can take a few minutes (torch is large).

**Verify:** `cd sam3_backend && uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"` → version + `True` on Apple Silicon.

## Step 4 — Start Label Studio

```bash
make up      # docker compose up -d (image heartexlabs/label-studio:1.23.2, :8080)
make logs    # follow logs until startup completes
```

**Verify:** open [http://localhost:8080](http://localhost:8080) and log in with `admin@example.com` / `admin12345`, or `curl -s http://localhost:8080/health`.

## Step 5 — Start the SAM3 backend

Background (normal use):

```bash
make backend-start
make backend-status   # running pid + {"status": "UP", ...} from /health
make backend-logs     # watch "SAM3 loaded in ...s" and the /health checks
```

Foreground (debugging): `make backend`.

First start: downloads a few GB of weights, loads SAM3 onto MPS (~1–3 min). Weights load **at startup** (`load_model()` in `_wsgi.py`), so token/licence problems fail here, loudly — check `logs/backend.log`.

**Verify from the container's point of view** (this is the exact path Label Studio uses):

```bash
docker exec label-studio curl -s http://host.docker.internal:9090/health
```

If this fails with a "reserved network address" error, see [Troubleshooting](/help/troubleshooting).

## Step 6 — Create the project

```bash
make setup
```

Under the hood (`scripts/setup_project.py`):

1. Waits up to ~2 min for Label Studio's `/health`.
2. Creates the project **"SAM3 annotation"** with `label_config.xml` (or reuses an existing one with that title).
3. Connects the ML backend: URL `http://host.docker.internal:9090`, title "SAM3 (MPS)", **interactive mode ON**.
4. Imports every `.jpg/.jpeg/.png/.webp/.bmp` in `./images/` as tasks.
5. Prints `Open http://localhost:8080/projects/<id>/data`.

Re-running `make setup` is safe: it reuses the project and backend connection and imports any new images. You can also import from the UI's **Import** button at any time.

**Verify:** in Label Studio, open the project → **Settings → Machine Learning** should list "SAM3 (MPS)" with state **Connected**.

::: tip "Test Request" returns `[]` — that's normal
SAM3 only works with an interactive click/box context, so a plain prediction request has nothing to segment. The backend logs `predict called without interactive context ...`. A **Connected** state is what matters.
:::

## Step 7 — Your first mask

Follow [Annotating with SAM3](/guide/annotating). The short version: **Auto-Annotation ON → magic tool → label → click → accept → Submit.**

## Step 8 — Verify end-to-end (optional)

```bash
make test
```

This exercises the same HTTP calls the UI makes: it uploads the COCO "two cats" photo, sends click and box prompts directly to the backend and through Label Studio's `interactive-annotating` path, checks mask IoU against ground-truth boxes (must exceed 0.75), measures cold vs. warm latency, creates annotations, and runs all exporters. Overlays and results land in `test_outputs/`.

For the real-UI test, see [Testing](/guide/testing).
