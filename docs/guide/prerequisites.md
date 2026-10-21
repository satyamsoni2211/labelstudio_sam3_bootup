# Prerequisites

Four things. The first one is the step everyone gets wrong — do it before anything else.

## 1. Hugging Face access to `facebook/sam3` (gated)

SAM3's weights are **gated**: the backend's first start downloads a few GB from Hugging Face and will fail with 401/403 without proper access.

1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and **accept the licence**.
2. Create a token that can read gated repos:
   - a classic **Read** token, or
   - a fine-grained token with **"Read access to contents of all public gated repos you can access"**.
3. Keep it for the `.env` step — it becomes `HF_TOKEN=...`. **Never commit it** (`.env` is git-ignored).

> Symptom if skipped: backend log shows 401/403 downloading `facebook/sam3`. Fix the token/licence, then restart the backend — no reinstall needed.

## 2. `uv`

[uv](https://docs.astral.sh/uv/) manages the backend environment. It installs the right Python itself — the backend requires **Python ≥3.11, <3.13** (`sam3_backend/pyproject.toml`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## 3. Docker Desktop (or compatible)

Label Studio runs as a container; you need `docker compose`.

```bash
docker --version
docker compose version
```

## 4. Apple Silicon Mac (recommended)

- **Apple Silicon** → PyTorch **MPS** (Apple GPU). This is the intended, fast path.
- Anything else falls back to **CPU** (slow) automatically.
- Override anytime with `DEVICE=cpu|mps|cuda` in the backend environment.

## Accounts and ports

| Thing | Value | Where it lives |
|---|---|---|
| Label Studio login | `admin@example.com` / `admin12345` | `.env` (`LS_USER_EMAIL`, `LS_PASSWORD`) |
| Label Studio API token | `LABEL_STUDIO_API_KEY` (40 hex chars) | `.env`, pre-seeded into the container |
| Label Studio | `http://localhost:8080` | Docker |
| SAM3 backend | `http://localhost:9090` (`ML_BACKEND_PORT`) | Native process |

Make sure ports **8080** and **9090** are free before you start. A stale backend holding 9090 is the most common "port in use" error — `make backend-stop` clears it.

## Disk and first-run expectations

- First backend start downloads the SAM3 weights (**a few GB**) into the Hugging Face cache, then loads them (~1–3 min total).
- `./ls-data/` holds all Label Studio state (projects, tasks, annotations). Deleting it deletes your projects.
- `./images/` is where you drop images to import; `./logs/backend.log` is where the backend explains itself.
