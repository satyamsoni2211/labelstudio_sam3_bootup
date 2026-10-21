#!/usr/bin/env python
"""Produce BOTH YOLO flavours from a Label Studio JSON export (brush / polygon / rectangle regions):

  <out>/yolo_box/{images,labels}/*.txt   "cls cx cy w h"        (detection; box = polygon bounding box)
  <out>/yolo_seg/{images,labels}/*.txt   "cls x1 y1 x2 y2 ..."  (segmentation; normalised polygon)
  each with classes.txt and data.yaml. Images are copied if found (see --images / LS download).

  python scripts/export_yolo.py --project 1 --out exports/yolo
  python scripts/export_yolo.py --export exports/project.json --images images/ --out exports/yolo
"""
import argparse
import json
import shutil
from pathlib import Path

import lsutil


def fetch_image(task_ref, s, url, images_dir):
    name = Path(task_ref.split("?")[0]).name
    if images_dir and (Path(images_dir) / name).exists():
        return Path(images_dir) / name
    if s and task_ref.startswith("/"):
        r = s.get(url + task_ref)
        if r.ok:
            p = Path("/tmp") / name
            p.write_bytes(r.content)
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export")
    ap.add_argument("--project", type=int)
    ap.add_argument("--images", help="directory with the original image files")
    ap.add_argument("--url", default=lsutil.LS_URL)
    ap.add_argument("--token", default=lsutil.LS_TOKEN)
    ap.add_argument("--label-config", default=str(lsutil.ROOT / "label_config.xml"))
    ap.add_argument("--out", default="exports/yolo")
    ap.add_argument("--rects", action="store_true",
                    help="also export hand-drawn RectangleLabels regions (by default only PolygonLabels; the "
                         "box_prompt rectangles are SAM prompts and would duplicate their polygons)")
    a = ap.parse_args()
    s = lsutil.ls_session(a.token)
    if a.project:
        r = s.get(f"{a.url}/api/projects/{a.project}/export", params={"exportType": "JSON", "download_all_tasks": "false"})
        r.raise_for_status()
        tasks = r.json()
    elif a.export:
        tasks = json.load(open(a.export))
    else:
        ap.error("give --export or --project")

    KINDS = lsutil.MASK_KINDS + (('rectanglelabels',) if a.rects else ())
    classes = lsutil.labels_from_config(a.label_config)
    out = Path(a.out)
    for sub in ("yolo_box", "yolo_seg"):
        (out / sub / "images").mkdir(parents=True, exist_ok=True)
        (out / sub / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for task in tasks:
        regions = list(lsutil.iter_regions(task, KINDS))
        if not regions:
            continue
        ref = next(iter(task["data"].values()))
        stem = lsutil.image_stem(ref)
        box_lines, seg_lines = [], []
        for res, label in regions:
            if label not in classes:
                classes.append(label)
            cid = classes.index(label)
            W, H = res["original_width"], res["original_height"]
            polys, (px0, py0, px1, py1), _ = lsutil.region_geometry(res, W, H)   # brush RLE -> contours
            x0, y0, x1, y1 = [min(max(c, 0), 1) for c in (px0 / W, py0 / H, px1 / W, py1 / H)]
            for p in polys:   # one seg line per contour of the instance
                seg_lines.append(f"{cid} " + " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in p))
            if not polys:     # rectangle
                seg_lines.append(f"{cid} {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}")
            box_lines.append(f"{cid} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
        img = fetch_image(ref, s, a.url, a.images)
        for sub, lines in (("yolo_box", box_lines), ("yolo_seg", seg_lines)):
            (out / sub / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            if img:
                shutil.copy(img, out / sub / "images" / f"{stem}{img.suffix}")
        n += 1
    for sub in ("yolo_box", "yolo_seg"):
        (out / sub / "classes.txt").write_text("\n".join(classes) + "\n")
        names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
        (out / sub / "data.yaml").write_text(
            f"path: {(out / sub).resolve()}\ntrain: images\nval: images\nnames:\n{names}\n")
    print(f"wrote YOLO box + seg labels for {n} images to {out}")


if __name__ == "__main__":
    main()
