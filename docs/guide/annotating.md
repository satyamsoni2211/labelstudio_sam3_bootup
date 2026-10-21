# Annotating with SAM3

Open [http://localhost:8080](http://localhost:8080) and the project **"SAM3 annotation"** (printed by `make setup`), then open a task.

## The 8 steps

### 1. Turn Auto-Annotation ON

Toggle in the **bottom bar**. With it off, the click/box tools only draw plain keypoints/rectangles and **never call SAM3**.

::: warning Per-browser state
The toggle state is stored **per browser**. A new browser, profile, or private window starts with it **off**. If "nothing happens", this is the first thing to check.
:::

### 2. Keep the label selected (recommended, once)

Gear icon in the bottom bar → **Keep label selected after creating a region**.

Without it, the label is deselected after every click and the refinement clicks in step 5 do nothing. (Re-picking the label before each click also works, but this setting ends the tedium.)

### 3. Pick the magic tool — first

Hover the **"Auto-Detect" wand** button at the bottom of the left-hand image toolbar (hotkey `M`) and choose:

- **Keypoint tool** — click prompts
- **Rectangle tool** — box prompts

::: tip Tool first, label second
Choose the tool *before* the label. Choosing the tool after the label deselects the label.
:::

### 4. Pick a label

Choose a label in the **Click prompt** row (e.g. `cat`) or the **Box prompt** row, matching the tool from step 3. The mask will take this label.

### 5. Click (or drag a box)

- **Click** the object, or **drag a box** around it.
- After ~0.2 s (about 1.5 s for the first prompt on an image — the embedding is computed once, then cached) SAM3's mask shows up as a **suggestion** and the bottom bar says "1 suggestions".
- **Refine** with more clicks; each click re-runs SAM3 with all points.
- **Alt/Option+click** adds a **negative point** (e.g. click the background or a neighbouring object to carve it out).

### 6. Accept or discard

- **Green check** in the bottom bar = accept. **Red cross** = discard.
- Or switch on **Auto-Accept Suggestions** to accept each result directly.

The mask becomes a normal `brush` region with the label you picked (or a polygon, with `SAM3_OUTPUT=polygon`).

### 7. Edit freely

Use the **brush / eraser** tools (top of the toolbar) to touch up edges, change the region's label, or delete it. The click/box prompt regions are only helpers — **exports ignore them**; only brush/polygon regions are exported.

### 8. Submit

Click **Submit** (or **Update** when editing an existing annotation).

## Behaviour worth knowing

- **Nothing happens / only a red dot after a click** → Auto-Annotation is off, or the magic tool isn't selected, or the label was deselected. Check `make backend-logs` for `interactive predict:` lines; none means the browser never called the backend.
- **Only the first click works** → enable "Keep label selected" (step 2) or re-pick the label before each click.
- **Old regions reappear in a task** → Label Studio restores autosaved drafts. Delete the draft or reload after submitting.
- **Slow first click on an image** → normal: the image embedding is computed once (~1.5 s on M4 Pro) and cached for 6 images (`SAM3_EMBED_CACHE`); later clicks take ~10–30 ms. The first request after a backend start also warms up MPS (~2 s).
- **Annotation events** (create/update/delete) are also sent to the backend and acknowledged/ignored — SAM3 is not trained. You'll see `LS event ... ignored` in the log. Expected.

## Click vs. box: which prompt?

| Situation | Prompt |
|---|---|
| Compact, well-separated object | Click — fastest |
| Large, elongated, or occluded object | Box — constrains the extent |
| Mask bleeds into a neighbour | Add an **Alt/Option+click** negative point on the neighbour |
| Mask misses a part | Add a positive click on the missed part |
