# Troubleshooting & FAQ

Start with the log that owns the problem: `make backend-logs` (SAM3 backend, `logs/backend.log`) or `make logs` (Label Studio container).

## Backend won't start / model download fails

### 401/403 downloading `facebook/sam3`

The token lacks gated-repo read permission, or the licence isn't accepted. Accept the licence at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and use a classic **Read** token (or fine-grained with "Read access to contents of all public gated repos you can access") in `.env` as `HF_TOKEN`. Details are in `logs/backend.log`. Then restart the backend — no reinstall needed.

### Unsupported MPS op errors

`PYTORCH_ENABLE_MPS_FALLBACK=1` is already set. Force CPU with `DEVICE=cpu`:

```bash
DEVICE=cpu scripts/backend_ctl.sh restart
```

### Stale backend process / port 9090 in use

```bash
make backend-stop
# or
kill $(cat logs/backend.pid)
```

Check state anytime with `make backend-status`.

## Label Studio can't reach the backend

### ML backend disconnected / "URL resolves to a reserved network address"

`host.docker.internal` resolves to a reserved address (e.g. `0.250.250.254`) that Label Studio refuses by default. `docker-compose.yml` sets `USE_DEFAULT_BANNED_SUBNETS=false` (and `SSRF_PROTECTION_ENABLED=false`) for this — recreate the container after editing:

```bash
make up
```

Test the exact path the container uses:

```bash
docker exec label-studio curl -s http://host.docker.internal:9090/health
```

### "Test Request" returns `[]`

**Normal.** SAM3 only works on a click/box (interactive context), so a plain prediction request has nothing to segment. The backend logs `predict called without interactive context ...`. A backend with state **Connected** is what matters — verify in Settings → Machine Learning, then annotate for real.

## API / token problems

### 401 from the LS API / scripts

Legacy API tokens are only enabled when the organization is created. `docker-compose.yml` sets `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN` and pre-seeds the token, so a **fresh** `./ls-data` is fine. If `ls-data` was created without them:

- **Clean fix (loses projects):** `make down && rm -rf ls-data && make up`, then `make setup`.
- **Keep data:** enable legacy tokens in **Organization → API Tokens Settings** in the UI.

### Backend cannot download the image

`LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` in `.env` must be valid **from the Mac**: `LABEL_STUDIO_URL=http://localhost:8080` and a token Label Studio accepts. The backend fetches every image through this API.

## Clicking does nothing (or almost nothing)

Work through this list — it covers nearly every report:

1. **Auto-Annotation ON?** Bottom-bar toggle. It's stored per browser; new browser/profile/private window = off.
2. **Magic tool selected?** Auto-Detect wand (hotkey `M`) → keypoint (clicks) or rectangle (boxes). Selected **before** the label.
3. **Label selected?** In the Click prompt / Box prompt row, after the tool.
4. **Backend receiving calls?** `make backend-logs` should show `interactive predict:` lines per click. None means the browser never called the backend — revisit 1–3, and confirm the backend is **Connected** in Settings → Machine Learning.

Related: **only the first click works** → enable **Keep label selected after creating a region** (gear icon, bottom bar) or re-pick the label before each click.

**Old regions reappear in a task** → Label Studio restores autosaved drafts. Delete the draft or reload after submitting.

## Performance

### Slow first click on an image

Expected: the image embedding is computed once (~1.5 s on M4 Pro) and cached for 6 images (`SAM3_EMBED_CACHE`); later clicks take ~10–30 ms. The first request after a backend start also warms up MPS (~2 s). Raise `SAM3_EMBED_CACHE` if you hop between many images.

## FAQ

**Does the backend train on my annotations?**
No. Annotation events are acknowledged and ignored (`LS event ... ignored`). SAM3 is inference-only here.

**Can I output polygons instead of brush masks?**
Yes — `SAM3_OUTPUT=polygon` (or `both`, which returns two overlapping regions per prompt — delete one). Tune with `SAM3_POLY_EPSILON` / `SAM3_POLY_MAX_POINTS`. See [Editing Labels](/guide/labels).

**Can I use Label Studio's built-in COCO/YOLO export?**
No — with `KeyPointLabels` in the config its COCO converter writes a single `"default"` category and its YOLO converter wrote empty label files (1.23.2). Use `make export-*`. Native `BRUSH_TO_PNG` / `BRUSH_TO_NUMPY` exports are fine.

**Where is my data?**
Label Studio state in `./ls-data/`, source images in `./images/`, backend log/PID in `logs/`, exports in `exports/`, model weights in the Hugging Face cache. Back up `ls-data/` + `images/` + `.env` to back up everything.

**Is it safe to expose port 8080?**
Not with the shipped compose settings (legacy token pre-seeded, SSRF checks off, default password). Keep it local, or change the password/token and revisit those settings first.
