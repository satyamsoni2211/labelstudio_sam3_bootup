#!/usr/bin/env python
"""End-to-end smoke test (needs `make up` + `make backend`, and HF access to facebook/sam3).

1. LS health + API token  2. project + image upload  3. connect backend (interactive)
4. direct POST /predict to the backend for a click and a box (LS wire format)
5. same through Label Studio's own path (/api/ml/{id}/interactive-annotating)
6. rasterise polygons, check coverage vs. known boxes, save overlays to test_outputs/
7. per-click latency (cold = embedding computed, warm = embedding cached)
8. create annotations from predictions; export COCO / YOLO / JSON via LS API; run the mask + yolo scripts
"""
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests

import lsutil

OUT = lsutil.ROOT / "test_outputs"
OUT.mkdir(exist_ok=True)
IMG = OUT / "cats.jpg"
URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
BACKEND_HOST = "http://localhost:9090"                   # from this Mac
BACKEND_FROM_LS = "http://host.docker.internal:9090"     # from the LS container
# approximate ground-truth boxes (x0,y0,x1,y1) of the two cats, eyeballed from the image
GT = {"left": (10, 55, 318, 474), "right": (345, 25, 640, 372)}
CLICK = {"left": (130, 250), "right": (460, 200)}
BOX = {"left": GT["left"], "right": GT["right"]}
results, fails = {}, []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info)
    if not cond:
        fails.append(name)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def raster(res, W, H):
    return lsutil.region_mask(res, W, H)   # brush RLE (or polygon) -> 0/1 mask


def overlay(m, name, color, extra=None):
    img = cv2.imread(str(IMG))
    img[m > 0] = (0.5 * img[m > 0] + 0.5 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, 2)
    if extra:
        extra(img)
    cv2.imwrite(str(OUT / name), img)


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    s = lsutil.ls_session()
    check("LS health", requests.get(lsutil.LS_URL + "/health").json().get("status") == "UP")
    r = s.get(lsutil.LS_URL + "/api/projects")
    check("LS API token (legacy) accepted", r.ok, r.text[:120] if not r.ok else "")
    if not r.ok:
        sys.exit(1)
    for old in s.get(lsutil.LS_URL + "/api/projects", params={"page_size": 200}).json().get("results", []):
        if old["title"].startswith("smoke-test-"):
            s.delete(f"{lsutil.LS_URL}/api/projects/{old['id']}")
    if not IMG.exists():
        IMG.write_bytes(requests.get(URL).content)
    H, W = cv2.imread(str(IMG)).shape[:2]

    cfg = (lsutil.ROOT / "label_config.xml").read_text()
    proj = s.post(lsutil.LS_URL + "/api/projects", json={"title": f"smoke-test-{int(time.time())}", "label_config": cfg}).json()
    pid = proj["id"]
    imp = s.post(f"{lsutil.LS_URL}/api/projects/{pid}/import", files={"file": ("cats.jpg", IMG.open("rb"))}).json()
    task_id = s.get(f"{lsutil.LS_URL}/api/tasks", params={"project": pid}).json()["tasks"][0]["id"]
    task = s.get(f"{lsutil.LS_URL}/api/tasks/{task_id}").json()
    print("project", pid, "task", task_id, task["data"])

    # --- direct backend /predict in LS wire format --------------------------------------
    bh = requests.get(BACKEND_HOST + "/health").json()
    check("backend /health", bh.get("status") == "UP", str(bh))

    def ctx_click(x, y, label="cat", pos=True):
        return {"id": "k1", "type": "keypointlabels", "from_name": "click_prompt", "to_name": "image", "is_positive": pos,
                "original_width": W, "original_height": H, "image_rotation": 0,
                "value": {"x": x / W * 100, "y": y / H * 100, "width": 0.5, "keypointlabels": [label]}}

    def ctx_box(b, label="cat"):
        return {"id": "r1", "type": "rectanglelabels", "from_name": "box_prompt", "to_name": "image",
                "original_width": W, "original_height": H, "image_rotation": 0,
                "value": {"x": b[0] / W * 100, "y": b[1] / H * 100, "width": (b[2] - b[0]) / W * 100,
                          "height": (b[3] - b[1]) / H * 100, "rotation": 0, "rectanglelabels": [label]}}

    # LS calls /setup (project id + label config) when it connects to a backend; do the same
    rs = requests.post(BACKEND_HOST + "/setup", json={"project": f"{pid}.{int(time.time())}", "schema": cfg,
                                                      "hostname": lsutil.LS_URL, "access_token": lsutil.LS_TOKEN})
    check("backend /setup", rs.ok, rs.text[:200])

    def direct(context_results):
        body = {"tasks": [task], "project": f"{pid}.{int(time.time())}", "label_config": cfg,
                "params": {"login": None, "password": None, "context": {"result": context_results}}}
        t0 = time.time()
        r = requests.post(BACKEND_HOST + "/predict", json=body)
        dt = (time.time() - t0) * 1000
        r.raise_for_status()
        j = r.json()
        return j["results"][0]["result"][0], j["results"][0].get("score"), dt

    lat = {}
    preds = {}
    for side in ("left", "right"):
        # 1st call for an image (cold) then repeat for warm latency
        res, score, dt = direct([ctx_click(*CLICK[side])])
        lat[f"click_{side}_first"] = dt
        res, score, dt2 = direct([ctx_click(*CLICK[side])])
        lat[f"click_{side}_warm"] = dt2
        m = raster(res, W, H)
        bb = bbox(m)
        area = int(m.sum())
        i = iou(bb, GT[side])
        check(f"direct click {side}: brush type/label/rle", res["type"] == "brushlabels" and res["value"]["brushlabels"] == ["cat"]
              and res["value"]["format"] == "rle" and res["from_name"] == "brush")
        check(f"direct click {side}: click inside mask", m[CLICK[side][1], CLICK[side][0]] == 1)
        check(f"direct click {side}: bbox IoU vs GT box > 0.75", i > 0.75, f"IoU={i:.3f} area={area}px ({area/(W*H)*100:.1f}%) score={score:.3f}")
        overlay(m, f"overlay_direct_click_{side}.png", (0, 255, 0), lambda im, c=CLICK[side]: cv2.circle(im, c, 6, (255, 255, 0), -1))
        results[f"click_{side}"] = {"area_px": area, "area_pct": round(area / (W * H) * 100, 2), "bbox_iou_gt": round(i, 3), "iou_score": round(score, 3), "rle_len": len(res["value"]["rle"])}
        preds[side] = res

        res, score, dt = direct([ctx_box(BOX[side])])
        lat[f"box_{side}"] = dt
        m = raster(res, W, H)
        i = iou(bbox(m), GT[side])
        area = int(m.sum())
        check(f"direct box {side}: bbox IoU vs GT box > 0.75", i > 0.75, f"IoU={i:.3f} area={area}px ({area/(W*H)*100:.1f}%) score={score:.3f}")
        overlay(m, f"overlay_direct_box_{side}.png", (0, 128, 255), lambda im, b=BOX[side]: cv2.rectangle(im, b[:2], b[2:], (255, 255, 0), 2))
        results[f"box_{side}"] = {"area_px": area, "bbox_iou_gt": round(i, 3), "iou_score": round(score, 3)}
        preds[side + "_box"] = res

    # click + negative click refinement: positive on left cat, negative on its tail
    res, score, dt = direct([ctx_click(*CLICK["left"]), {**ctx_click(180, 310, pos=False), "id": "k2"}])
    lat["multi_click"] = dt
    m = raster(res, W, H)
    check("direct pos+neg click returns brush mask", m.sum() > 1000, f"area={int(m.sum())}")
    overlay(m, "overlay_direct_pos_neg.png", (255, 0, 255))

    # --- cold vs warm latency on never-seen images (mirrored copies so the embedding cache cannot hit) ---
    cold, warm = [], []
    for k in range(3):
        fimg = cv2.flip(cv2.imread(str(IMG)), 1) if k % 2 == 0 else cv2.imread(str(IMG))[:, :, ::-1].copy()
        fimg = cv2.resize(fimg, (W - 8 * (k + 1), H - 6 * (k + 1)))  # unique pixels -> unique upload/hash
        fp = OUT / f"_latency_{k}.jpg"
        cv2.imwrite(str(fp), fimg)
        up = s.post(f"{lsutil.LS_URL}/api/projects/{pid}/import", files={"file": (fp.name, fp.open("rb"))})
        tid = s.get(f"{lsutil.LS_URL}/api/tasks", params={"project": pid, "page_size": 50}).json()["tasks"]
        tid = max(t["id"] for t in tid)
        tk = s.get(f"{lsutil.LS_URL}/api/tasks/{tid}").json()
        h2, w2 = fimg.shape[:2]
        c = {**ctx_click(w2 * 0.3, h2 * 0.5), "original_width": w2, "original_height": h2}
        for n in range(4):
            body = {"tasks": [tk], "project": f"{pid}.0", "label_config": cfg, "params": {"context": {"result": [c]}}}
            t0 = time.time()
            rr = requests.post(BACKEND_HOST + "/predict", json=body); rr.raise_for_status()
            dt = (time.time() - t0) * 1000
            (cold if n == 0 else warm).append(dt)
        fp.unlink()
    lat["cold_click_new_image_median"] = float(np.median(cold))
    lat["cold_click_new_image_all"] = cold
    lat["warm_click_median"] = float(np.median(warm))
    lat["warm_click_max"] = float(np.max(warm))

    via_ls_result = None
    # --- through Label Studio's own path -------------------------------------------------
    r = s.post(lsutil.LS_URL + "/api/ml", json={"project": pid, "url": BACKEND_FROM_LS, "title": "SAM3", "is_interactive": True})
    check("LS accepts ML backend (validation via host.docker.internal)", r.ok, r.text[:300])
    if r.ok:
        ml = r.json()
        st = s.get(f"{lsutil.LS_URL}/api/ml/{ml['id']}").json()
        check("LS /api/ml state is Connected", st.get("state") == "CO", f"state={st.get('state')} is_interactive={st.get('is_interactive')}")
        t0 = time.time()
        r = s.post(f"{lsutil.LS_URL}/api/ml/{ml['id']}/interactive-annotating", json={"task": task_id, "context": {"result": [ctx_click(*CLICK["right"])]}})
        lat["via_LS_click_right"] = (time.time() - t0) * 1000
        ok = r.ok and r.json().get("data", {}).get("result")
        check("LS interactive-annotating returns result", bool(ok), r.text[:300])
        if ok:
            res = r.json()["data"]["result"][0]
            m = raster(res, W, H)
            i = iou(bbox(m), GT["right"])
            check("via LS: click right cat bbox IoU > 0.75", i > 0.75, f"IoU={i:.3f} area={int(m.sum())}px")
            overlay(m, "overlay_via_LS_click_right.png", (0, 255, 255))
            via_ls_result = res

    # --- annotation from predictions (one from the LS interactive path, one from the direct call) -----------
    result = []
    for p in (preds["left_box"], via_ls_result or preds["right_box"]):
        p = dict(p); p.pop("score", None); p.pop("readonly", None)
        result.append(p)
    r = s.post(f"{lsutil.LS_URL}/api/tasks/{task_id}/annotations", json={"result": result, "ground_truth": False})
    check("annotation created from SAM3 predictions", r.ok, r.text[:100])

    exp = OUT / "ls_exports"
    # Label Studio's own PNG brush export must work on these annotations
    r = s.get(f"{lsutil.LS_URL}/api/projects/{pid}/export", params={"exportType": "BRUSH_TO_PNG", "download_all_tasks": "false", "ids[]": [task_id]})
    if r.ok:
        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        check("LS native BRUSH_TO_PNG export has PNG masks", sum(n.endswith(".png") for n in names) >= 2, str(names[:4]))
    else:
        check("LS native BRUSH_TO_PNG export", False, f"HTTP {r.status_code} {r.text[:150]}")
    shutil.rmtree(exp, ignore_errors=True)
    exp.mkdir()
    for fmt in ("JSON", "COCO", "YOLO"):   # LS built-in exports (informational, see README: KeyPointLabels confuse the converters)
        r = s.get(f"{lsutil.LS_URL}/api/projects/{pid}/export", params={"exportType": fmt, "download_all_tasks": "false", "ids[]": [task_id]})
        if not r.ok:
            print(f"LS export {fmt}: HTTP {r.status_code} {r.text[:120]}")
            continue
        if fmt == "JSON":
            (exp / "project.json").write_bytes(r.content)
        else:
            d = exp / ("LS_" + fmt)
            d.mkdir(exist_ok=True)
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(d)
        print(f"LS export {fmt}: ok ({len(r.content)} bytes)")
    py = sys.executable
    sc = lsutil.ROOT / "scripts"
    P = str(exp / "project.json")
    subprocess.check_call([py, str(sc / "export_coco.py"), "--export", P, "--out", str(exp / "coco" / "result.json")])
    subprocess.check_call([py, str(sc / "export_masks.py"), "--export", P, "--out", str(exp / "masks")])
    subprocess.check_call([py, str(sc / "export_yolo.py"), "--export", P, "--images", str(OUT), "--out", str(exp / "yolo")])
    c = json.load(open(exp / "coco" / "result.json"))
    check("COCO (from brush): 2 annotations, polygon segmentation, category 'cat'", len(c["annotations"]) == 2 and all(a["segmentation"] for a in c["annotations"])
          and {x["name"] for x in c["categories"]} >= {"cat"} and all(a["category_id"] == 1 for a in c["annotations"]),
          f"bbox={[a['bbox'] for a in c['annotations']]} area={[round(a['area']) for a in c['annotations']]} seg_pts={[len(a['segmentation'][0])//2 for a in c['annotations']]}")
    for gt, a_ in zip((GT["left"], GT["right"]), c["annotations"]):
        bb = a_["bbox"]
        check("COCO bbox IoU vs GT > 0.9", iou((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]), gt) > 0.9, f"bbox={bb}")
    for sub in ("yolo_box", "yolo_seg"):
        lines = (next((exp / "yolo" / sub / "labels").glob("*.txt"))).read_text().strip().splitlines()
        check(f"{sub}: 2 lines, class 0", len(lines) == 2 and all(l.split()[0] == "0" for l in lines), " | ".join(l[:70] for l in lines))
    # the polygon path must still work: convert a brush mask to a polygon region and export that too
    sys.path.insert(0, str(lsutil.ROOT / "sam3_backend"))
    from sam3_model import mask_to_polygon
    mL = raster(preds["left_box"], W, H)
    poly = mask_to_polygon(mL)
    check("polygon output path: mask_to_polygon works", poly is not None and len(poly) >= 8, f"points={len(poly or [])}")
    pj = json.load(open(P))
    pres = {"id": "p1", "type": "polygonlabels", "from_name": "polygon", "to_name": "image", "original_width": W, "original_height": H,
            "image_rotation": 0, "value": {"points": poly, "closed": True, "polygonlabels": ["cat"]}}
    pj[0]["annotations"][-1]["result"] = [pres]
    json.dump(pj, open(exp / "poly_project.json", "w"))
    subprocess.check_call([py, str(sc / "export_coco.py"), "--export", str(exp / "poly_project.json"), "--out", str(exp / "coco_poly" / "result.json")])
    cp = json.load(open(exp / "coco_poly" / "result.json"))["annotations"][0]
    check("polygon COCO area close to brush area", abs(cp["area"] - mL.sum()) / mL.sum() < 0.03, f"poly={cp['area']:.0f} brush={int(mL.sum())}")
    inst = cv2.imread(str(next((exp / "masks").glob("*_instance.png"))), cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(str(next((exp / "masks").glob("*_semantic.png"))), cv2.IMREAD_UNCHANGED)
    check("instance mask has ids {0,1,2}", set(np.unique(inst)) <= {0, 1, 2} and {1, 2} <= set(np.unique(inst)), f"unique={np.unique(inst).tolist()} dtype={inst.dtype}")
    check("semantic mask has classes {0,1}", set(np.unique(sem)) == {0, 1} and sem.dtype == np.uint8, f"unique={np.unique(sem).tolist()} px={int((sem>0).sum())}")

    print("\nLATENCY (ms, wall-clock incl. HTTP, image fetch/decode, polygon conversion):")
    for k, v in lat.items():
        print(f"  {k:30s} " + (f"{v:8.0f}" if not isinstance(v, list) else str([round(x) for x in v])))
    (OUT / "smoke_results.json").write_text(json.dumps({"results": results, "latency_ms": {k: (round(v) if not isinstance(v, list) else [round(x) for x in v]) for k, v in lat.items()}, "fails": fails}, indent=1))
    print("\nFAILED:" if fails else "\nALL CHECKS PASSED", fails or "")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
