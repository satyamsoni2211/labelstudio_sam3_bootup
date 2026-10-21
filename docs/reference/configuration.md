# Configuration & Environment

All configuration is via `.env` (copied from `.env.example`) plus backend process environment variables. Never commit `.env`.

## `.env` variables

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `HF_TOKEN` | — (required) | SAM3 backend | Hugging Face token with gated read access to `facebook/sam3` |
| `LS_USER_EMAIL` | `admin@example.com` | Container (first start) | Admin login |
| `LS_PASSWORD` | `admin12345` | Container (first start) | Admin password |
| `LABEL_STUDIO_API_KEY` | `0123…4567` (40 hex) | Backend + all scripts | Legacy API token; pre-seeded via `LABEL_STUDIO_USER_TOKEN` |
| `LABEL_STUDIO_URL` | `http://localhost:8080` | Backend + scripts (host side) | Where the Mac reaches Label Studio; backend downloads images through this API |
| `ML_BACKEND_PORT` | `9090` | Backend | Listening port |
| `SAM3_MODEL_ID` | `facebook/sam3` | Backend | Hugging Face model ID |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Backend | Unsupported MPS ops fall back to CPU instead of erroring |

## Backend-only variables

Set these in the backend process environment (not read from `docker-compose.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | auto (MPS → CUDA → CPU) | Force `cpu`, `mps`, or `cuda` |
| `SAM3_OUTPUT` | `brush` | `brush` \| `polygon` \| `both` — region type returned per prompt |
| `SAM3_EMBED_CACHE` | `6` | Images whose embeddings are cached (LRU); first prompt per image ~1.5 s, then ~10–30 ms |
| `SAM3_POLY_EPSILON` | `0.0015` | Polygon simplification (fraction of contour perimeter; smaller = more vertices) |
| `SAM3_POLY_MAX_POINTS` | `400` | Max polygon vertices |
| `MODEL_DIR` | `sam3_backend/model_dir` | SDK job-manager directory; must be set so annotation events don't crash it |

Example — polygon output with finer detail:

```bash
SAM3_OUTPUT=polygon SAM3_POLY_EPSILON=0.0008 scripts/backend_ctl.sh restart
```

## `label_config.xml`

Four label groups (`brush`, `click_prompt`, `box_prompt`, `polygon`) that must carry identical `<Label>` lists — see [Editing Labels](/guide/labels). Apply changes with:

```bash
cd sam3_backend && uv run python ../scripts/setup_project.py --update-config
```

## `docker-compose.yml` essentials

| Setting | Value | Why |
|---|---|---|
| Image | `heartexlabs/label-studio:1.23.2` | Pinned Label Studio version |
| Ports | `8080:8080` | UI + API |
| `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN` | `true` | Legacy tokens only work when enabled at org creation |
| `USE_DEFAULT_BANNED_SUBNETS` / `SSRF_PROTECTION_ENABLED` | `false` | Let LS register `host.docker.internal` (reserved address) as an ML backend |
| `LOCAL_FILES_SERVING_ENABLED` + volume `./images:/label-studio/files/images` | on | Serve local images |
| Volumes | `./ls-data:/label-studio/data` | All LS state persists here |
| Healthcheck | `curl -fsS http://localhost:8080/health` | `make setup` waits on this |

::: danger Local only
The token and SSRF settings suit a **local-only** instance. Do not expose port 8080 publicly with them. Change the default password and API token before any shared use.
:::
