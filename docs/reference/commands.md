# Commands (Makefile)

Everyday workflow is `make`-driven from the repository root.

## Lifecycle

| Command | What it does |
|---|---|
| `make sync` | Install backend dependencies (`cd sam3_backend && uv sync`) — run once |
| `make up` | Start Label Studio (Docker) on `http://localhost:8080` |
| `make down` | Stop Label Studio (data stays in `./ls-data`) |
| `make logs` | Follow Label Studio logs |

## SAM3 backend

| Command | What it does |
|---|---|
| `make backend` | Run the backend in the **foreground** (MPS) on `:9090` |
| `make backend-start` | Run it in the **background** (`logs/backend.log`, `logs/backend.pid`) |
| `make backend-stop` | Stop the background backend |
| `make backend-status` | Running or not + `curl localhost:9090/health` |
| `make backend-logs` | `tail -f logs/backend.log` |
| `scripts/backend_ctl.sh restart` | Stop + start (use to apply env changes like `SAM3_OUTPUT=...`) |

## Project

| Command | What it does |
|---|---|
| `make setup` | Create/reuse project "SAM3 annotation", connect the backend, import `./images/*` |

## Tests

| Command | What it does |
|---|---|
| `make test` | End-to-end smoke test (needs `make up` + running backend) → `test_outputs/` |
| `make ui-test` | Drive the real LS UI headlessly (Playwright). Deletes annotations/drafts of the first task of project 6 — `UI_PROJECT=<id>` to change. One-time: `(cd sam3_backend && uv run playwright install chromium)` |

See [Testing](/guide/testing) for what each test asserts.

## Exports

`PROJECT` = project ID from the URL (`make setup` prints it), `OUT` = output dir (default `exports/`).

| Command | Output |
|---|---|
| `make export-json PROJECT=1` | `exports/project.json` (raw LS JSON) |
| `make export-coco PROJECT=1` | `exports/coco/result.json` |
| `make export-yolo PROJECT=1` | `exports/yolo/yolo_box/` + `yolo_seg/` |
| `make export-masks PROJECT=1` | `exports/masks/` (semantic + instance PNGs) |
| `make export-all PROJECT=1` | All of the above |

See [Exporting Annotations](/guide/exports).
