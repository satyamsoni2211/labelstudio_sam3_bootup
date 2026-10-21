#!/usr/bin/env python
"""COCO instance-segmentation export (polygons + bounding boxes) from a Label Studio JSON export.

Why not Label Studio's built-in COCO export? With KeyPointLabels in the config LS's converter treats the
project as a keypoint task and writes a single category "default". This script uses the PolygonLabels / BrushLabels
regions (brush RLE is converted to polygons) and the label order of label_config.xml (category id = index + 1).

  python scripts/export_coco.py --project 1 --out exports/coco/result.json
  python scripts/export_coco.py --export exports/project.json --out exports/coco/result.json
"""
import argparse
import json
from pathlib import Path

import lsutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export")
    ap.add_argument("--project", type=int)
    ap.add_argument("--url", default=lsutil.LS_URL)
    ap.add_argument("--token", default=lsutil.LS_TOKEN)
    ap.add_argument("--label-config", default=str(lsutil.ROOT / "label_config.xml"))
    ap.add_argument("--out", default="exports/coco/result.json")
    ap.add_argument("--rects", action="store_true", help="also export hand-drawn RectangleLabels (box only, no segmentation)")
    a = ap.parse_args()
    if a.project:
        r = lsutil.ls_session(a.token).get(f"{a.url}/api/projects/{a.project}/export",
                                           params={"exportType": "JSON", "download_all_tasks": "false"})
        r.raise_for_status()
        tasks = r.json()
    elif a.export:
        tasks = json.load(open(a.export))
    else:
        ap.error("give --export or --project")

    kinds = lsutil.MASK_KINDS + (("rectanglelabels",) if a.rects else ())
    classes = lsutil.labels_from_config(a.label_config)
    images, anns = [], []
    for task in tasks:
        regions = list(lsutil.iter_regions(task, kinds))
        if not regions:
            continue
        ref = next(iter(task["data"].values()))
        W, H = regions[0][0]["original_width"], regions[0][0]["original_height"]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": lsutil.image_stem(ref) + Path(ref.split("?")[0]).suffix,
                       "width": W, "height": H})
        for res, label in regions:
            if label not in classes:
                classes.append(label)
            polys, (x0, y0, x1, y1), area = lsutil.region_geometry(res, W, H)   # brush RLE -> contours
            seg = [[round(float(c), 2) for c in p.reshape(-1)] for p in polys]
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": classes.index(label) + 1,
                         "segmentation": seg, "bbox": [round(float(x0), 2), round(float(y0), 2),
                                                       round(float(x1 - x0), 2), round(float(y1 - y0), 2)],
                         "area": round(float(area), 2), "iscrowd": 0})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": images, "categories": [{"id": i + 1, "name": c, "supercategory": "object"}
                                                                for i, c in enumerate(classes)],
                               "annotations": anns}, indent=1))
    print(f"wrote {len(anns)} annotations on {len(images)} images to {out}")


if __name__ == "__main__":
    main()
