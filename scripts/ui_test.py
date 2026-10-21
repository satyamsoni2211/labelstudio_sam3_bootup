#!/usr/bin/env python
"""Drive the REAL Label Studio UI headlessly (Playwright/Chromium) and check that SAM3 masks appear.

Steps (screenshots in test_outputs/ui_*.png):
  login -> open the project's first task -> Auto-Annotation ON -> select the click-prompt label "cat" ->
  click the right cat -> a brush region must appear (and the backend log must show an interactive /predict) ->
  select the box-prompt label "cat" -> drag a box around the left cat -> second brush region ->
  Submit -> Update -> no ERROR/Traceback in the backend log -> check the stored annotation through the API ->
  run the COCO / YOLO / PNG-mask exports on it and check that the masks sit on the right cats.

  cd sam3_backend && uv run python ../scripts/ui_test.py [--project 6] [--headed]

WARNING: it deletes the existing annotations of the task it uses (default project 6, first task).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

import lsutil

ROOT = lsutil.ROOT
OUT = ROOT / "test_outputs"
LOG = ROOT / "logs" / "backend.log"
LS = lsutil.LS_URL
EMAIL = lsutil.os.getenv("LS_USER_EMAIL", "admin@example.com")
PASSWORD = lsutil.os.getenv("LS_PASSWORD", "admin12345")
# image fractions: right cat (click) and left cat (box), from the COCO cats photo (640x480)
CLICK_FRAC = (0.75, 0.45)
BOX_FRAC = (10 / 640 + 0.02, 55 / 480 + 0.03, 318 / 640 - 0.02, 474 / 480 - 0.03)
GT = {"right": (345, 25, 640, 372), "left": (10, 55, 318, 474)}
fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info)
    if not cond:
        fails.append(name)


def log_text():
    return LOG.read_text(errors="replace") if LOG.exists() else ""


def n_interactive_predicts():
    return len(re.findall(r"interactive predict:", log_text()))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def region_types(page):
    """Text of each item in the Regions panel."""
    return page.evaluate("""() => [...document.querySelectorAll('[class*="outliner-item"], [class*="OutlinerItem"]')]
        .filter(e => e.className.toString().match(/outliner-item(__|-)?(content|title)?$|outliner-item\\b/) && !e.className.toString().includes('__'))
        .map(e => e.innerText.replace(/\\s+/g, ' ').trim())""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=6)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    s = lsutil.ls_session()
    tasks = s.get(f"{LS}/api/tasks", params={"project": a.project, "page_size": 5}).json()["tasks"]
    assert tasks, f"project {a.project} has no tasks (run `make setup`)"
    task_id = tasks[0]["id"]
    for an in s.get(f"{LS}/api/tasks/{task_id}/annotations").json():
        s.delete(f"{LS}/api/annotations/{an['id']}")
    for d in s.get(f"{LS}/api/tasks/{task_id}/drafts").json():   # LS autosaves drafts and would restore old regions
        s.delete(f"{LS}/api/drafts/{d['id']}")
    log_start = len(log_text())
    print(f"project {a.project}, task {task_id}; cleared old annotations and drafts")

    api_calls, console_errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not a.headed)
        page = browser.new_context(viewport={"width": 1600, "height": 1500}).new_page()   # fresh profile: toggle starts OFF
        page.on("request", lambda r: api_calls.append((r.method, r.url, r.post_data)) if "/api/ml/" in r.url else None)
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        # 1. login
        page.goto(f"{LS}/user/login", wait_until="domcontentloaded")
        page.fill('input[name=email]', EMAIL)
        page.fill('input[name=password]', PASSWORD)
        page.click('button[type=submit]')
        for _ in range(60):   # the app navigates away from /user/login once the session cookie is set
            if "/user/login" not in page.url:
                break
            page.wait_for_timeout(500)
        check("logged in", "/user/login" not in page.url, page.url)
        # 2. open the task
        page.goto(f"{LS}/projects/{a.project}/data?task={task_id}", wait_until="domcontentloaded")
        page.wait_for_selector("canvas", timeout=30000)
        page.wait_for_selector('[data-testid="bottombar-auto-annotation-toggle"]', state="attached", timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "ui_01_task_open.png"))
        toggle = page.locator('[data-testid="bottombar-auto-annotation-toggle"]')
        check("Auto-Annotation starts OFF in a fresh browser", not toggle.is_checked())

        # 3. Auto-Annotation ON
        toggle.locator("xpath=..").click()
        page.wait_for_timeout(500)
        if not toggle.is_checked():
            toggle.check(force=True)
        check("Auto-Annotation toggled ON", toggle.is_checked())
        page.screenshot(path=str(OUT / "ui_02_auto_annotation_on.png"))

        labels = page.locator(".lsf-label")
        names = [t.strip() for t in labels.all_inner_texts()]
        print("label buttons:", [re.sub(r"\s+", " ", n) for n in names])
        check("12 label buttons (brush/click/box/polygon x cat,dog,person)", len(names) == 12)
        # the image stage canvas (the first <canvas> is a larger overlay): same aspect ratio as the 640x480 photo
        canvas = next(c for c in page.locator("canvas").all() if abs(c.bounding_box()["width"] / c.bounding_box()["height"] - 640 / 480) < 0.01)

        # LS setting: "Keep label selected after creating a region" (otherwise every click needs the label re-picked)
        page.locator('[data-testid="bottombar-settings-button"]').click()
        page.get_by_text("Keep label selected after creating a region").first.click()
        page.wait_for_timeout(300)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "ui_02b_settings_keep_label.png"))

        def img_xy(fx, fy):
            bb = canvas.bounding_box()
            return bb["x"] + fx * bb["width"], bb["y"] + fy * bb["height"]

        def select_magic_tool(kind):
            """Toolbar: hover the magic 'Auto-Detect' button (bottom), pick the smart keypoint / rectangle tool.
            The smart (dynamic) tool must be chosen BEFORE the label, otherwise picking the label switches back."""
            page.locator("button.lsf-tool_smart").first.hover()
            page.wait_for_timeout(500)
            page.locator(f"button.lsf-tool_smart[aria-label={kind}]").click()
            page.wait_for_timeout(300)

        def wait_predict(before, what):
            t0 = time.time()
            while time.time() - t0 < 60 and n_interactive_predicts() == before:
                page.wait_for_timeout(200)
            ok = n_interactive_predicts() > before
            check(f"backend log shows an interactive /predict after the {what}", ok, f"{time.time() - t0:.1f}s")
            page.wait_for_selector('[data-testid="bottombar-accept-suggestions-button"]', timeout=20000)   # the mask arrives as a suggestion
            page.wait_for_timeout(800)

        # 4. click prompt on the right cat (magic keypoint tool + label "cat" of the click_prompt row)
        before = n_interactive_predicts()
        n_ia = len([c for c in api_calls if "interactive-annotating" in c[1]])
        select_magic_tool("key-point-tool")
        labels.nth(3).click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "ui_03_magic_tool_and_label_selected.png"))
        x, y = img_xy(*CLICK_FRAC)
        page.mouse.move(x, y)
        page.mouse.down(); page.mouse.up()
        wait_predict(before, "click")
        page.screenshot(path=str(OUT / "ui_04_click_suggestion.png"))
        ia = [c for c in api_calls if "interactive-annotating" in c[1]]
        check("browser sent POST /api/ml/<id>/interactive-annotating", len(ia) > n_ia and ia[-1][0] == "POST", f"{len(ia)} request(s)")
        print("   request body:", (ia[-1][2] or "")[:260] if ia else None)

        # 4b. refine with a negative click (Alt/Option+click) on the pink blanket -> second request with 2 points
        before = n_interactive_predicts()
        nx, ny = img_xy(0.62, 0.93)
        page.keyboard.down("Alt")
        page.mouse.move(nx, ny)
        page.mouse.down(); page.mouse.up()
        page.keyboard.up("Alt")
        wait_predict(before, "negative (Alt) click")
        last = [l for l in log_text().splitlines() if "interactive predict:" in l][-1]
        check("refinement request carried 2 points (1 positive + 1 negative)", "2 point(s)" in last, last[-110:])
        page.screenshot(path=str(OUT / "ui_05_after_negative_click.png"))

        # 4c. accept the suggestion -> a real brush region
        page.locator('[data-testid="bottombar-accept-suggestions-button"]').click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "ui_06_click_mask_accepted.png"))
        regs = region_types(page)
        print("regions panel:", regs)
        types = page.evaluate("window.Htx.annotationStore.selected.regions.map(r => r.type)")
        check("accepted mask is a brush region", "brushregion" in types, str(types))

        # 5. box prompt on the left cat (magic rectangle tool + label "cat" of the box_prompt row)
        before = n_interactive_predicts()
        select_magic_tool("rectangle-tool")
        labels.nth(6).click()
        page.wait_for_timeout(300)
        x0, y0 = img_xy(BOX_FRAC[0], BOX_FRAC[1])
        x1, y1 = img_xy(BOX_FRAC[2], BOX_FRAC[3])
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=8)
        page.mouse.move(x1, y1, steps=8)
        page.mouse.up()
        wait_predict(before, "box")
        page.screenshot(path=str(OUT / "ui_07_box_suggestion.png"))
        page.locator('[data-testid="bottombar-accept-suggestions-button"]').click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "ui_08_both_masks_accepted.png"))
        types = page.evaluate("window.Htx.annotationStore.selected.regions.map(r => r.type)")
        print("regions panel:", region_types(page), types)
        check("two brush regions after click + box", types.count("brushregion") == 2, str(types))

        # 6. submit the annotation
        page.locator('[data-testid="bottombar-submit-button"]').click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "ui_09_submitted.png"))
        check("Update button visible after Submit (annotation saved)", page.locator('[data-testid="bottombar-update-button"]').count() == 1)

        # 7. edit the submitted annotation: one more click prompt (label "dog" on the left remote control), accept, Update
        before = n_interactive_predicts()
        select_magic_tool("key-point-tool")
        labels.nth(4).click()
        page.mouse.move(*img_xy(0.15, 0.185))
        page.mouse.down(); page.mouse.up()
        wait_predict(before, "click on the remote control")
        page.locator('[data-testid="bottombar-accept-suggestions-button"]').click()
        page.wait_for_timeout(1500)
        page.wait_for_function("() => !document.querySelector('[data-testid=\"bottombar-update-button\"]')?.disabled", timeout=10000)
        page.screenshot(path=str(OUT / "ui_10_third_mask_before_update.png"))
        page.locator('[data-testid="bottombar-update-button"]').click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "ui_11_updated.png"))
        check("no app console errors mentioning interactive/ML", not [e for e in console_errors if "interactive" in e.lower() or "/api/ml" in e.lower()],
              str(console_errors[:3]))
        browser.close()

    # 7. backend log must be clean
    new_log = log_text()[log_start:]
    bad = [l for l in new_log.splitlines() if re.search(r"Traceback|Error|ERROR| 500 ", l)]
    check("no errors / tracebacks / HTTP 500 in backend log during the UI run", not bad, "; ".join(bad[:3]))
    print("backend log lines of interest:")
    for l in new_log.splitlines():
        if "interactive predict" in l or "LS event" in l or "/webhook" in l or "SAM3 inference" in l:
            print("   ", l[-160:])

    # 8. stored annotation
    anns = s.get(f"{LS}/api/tasks/{task_id}/annotations").json()
    check("annotation stored", len(anns) >= 1)
    res = anns[-1]["result"] if anns else []
    kinds = [r["type"] for r in res]
    print("stored result types:", kinds)
    brushes = [r for r in res if r["type"] == "brushlabels"]
    cats = [r for r in brushes if r["value"]["brushlabels"] == ["cat"]]
    check("annotation holds 3 brushlabels regions (2 cat + 1 dog)", len(brushes) == 3 and len(cats) == 2, str(kinds))
    if len(cats) == 2:
        W, H = brushes[0]["original_width"], brushes[0]["original_height"]
        boxes = [bbox(lsutil.region_mask(r, W, H)) for r in cats]
        sc = (640 / W, 480 / H)
        for name in ("right", "left"):
            best = max(iou(tuple(c * k for c, k in zip(b, sc * 2)), GT[name]) for b in boxes)
            check(f"a stored mask covers the {name} cat (bbox IoU > 0.7)", best > 0.7, f"IoU={best:.3f}")

    # 9. exports
    ex = OUT / "ui_exports"
    py = sys.executable
    sc_ = ROOT / "scripts"
    for cmd in ([sc_ / "export_coco.py", "--out", ex / "coco" / "result.json"],
                [sc_ / "export_yolo.py", "--out", ex / "yolo"],
                [sc_ / "export_masks.py", "--out", ex / "masks"]):
        subprocess.check_call([py, *map(str, cmd), "--project", str(a.project)])
    coco = json.load(open(ex / "coco" / "result.json"))
    check("COCO export: 3 annotations with segmentation", len(coco["annotations"]) == 3 and all(x["segmentation"] for x in coco["annotations"]),
          str([(x["bbox"], round(x["area"])) for x in coco["annotations"]]))
    for sub in ("yolo_box", "yolo_seg"):
        lines = next((ex / "yolo" / sub / "labels").glob("*.txt")).read_text().split("\n")
        check(f"YOLO {sub}: >= 3 lines", len([l for l in lines if l]) >= 3)
    sem = cv2.imread(str(next((ex / "masks").glob("*_semantic.png"))), cv2.IMREAD_UNCHANGED)
    check("semantic PNG has classes 0,1(cat),2(dog)", set(np.unique(sem)) == {0, 1, 2}, f"unique={np.unique(sem).tolist()} px={int((sem > 0).sum())}")
    r = s.get(f"{LS}/api/projects/{a.project}/export", params={"exportType": "BRUSH_TO_PNG", "download_all_tasks": "false", "ids[]": [task_id]})
    import io, zipfile
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist() if r.ok else []
    check("LS native BRUSH_TO_PNG export", sum(n.endswith(".png") for n in names) >= 3, str(names[:3]))

    # overlay of the stored masks for the record
    img_path = OUT / "cats.jpg"
    if img_path.exists() and len(brushes) >= 1:
        img = cv2.imread(str(img_path))
        for r_, col in zip(brushes, ((0, 255, 0), (0, 128, 255), (255, 0, 255))):
            m = cv2.resize(lsutil.region_mask(r_, W, H), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img[m > 0] = (0.5 * img[m > 0] + 0.5 * np.array(col)).astype(np.uint8)
        cv2.imwrite(str(OUT / "ui_08_stored_masks_overlay.png"), img)

    print("\nFAILED: " + ", ".join(fails) if fails else "\nALL UI CHECKS PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
