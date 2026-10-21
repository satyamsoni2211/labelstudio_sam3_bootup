#!/usr/bin/env python
"""Write per-image PNG masks from a Label Studio JSON export (brush / polygon / rectangle regions).

Modes (--mode):
  semantic : <stem>_semantic.png  8-bit, 0 = background, class_id+1 = pixels of that class
             (class ids follow the label order in label_config.xml; overlapping objects: later wins)
  instance : <stem>_instance.png  16-bit, 0 = background, 1..N = one id per object (annotation order)
             plus <stem>_instance.json mapping instance id -> label
  both     : both of the above (default)
Also writes classes.json (legend). Image size comes from the annotation (original_width/height),
so no image files are needed. Use --rects to also rasterise RectangleLabels regions as filled boxes.

Input: a JSON export file (Export -> JSON, or `make export-json`), OR --project N to pull it from
the Label Studio API using LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY from .env.

  python scripts/export_masks.py --export exports/project.json --out exports/masks
  python scripts/export_masks.py --project 1 --out exports/masks --mode semantic
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import lsutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", help="LS JSON export file")
    ap.add_argument("--project", type=int, help="pull JSON export from this LS project id instead")
    ap.add_argument("--url", default=lsutil.LS_URL)
    ap.add_argument("--token", default=lsutil.LS_TOKEN)
    ap.add_argument("--label-config", default=str(lsutil.ROOT / "label_config.xml"))
    ap.add_argument("--out", default="exports/masks")
    ap.add_argument("--mode", choices=["semantic", "instance", "both"], default="both")
    ap.add_argument("--rects", action="store_true", help="also rasterise rectangle regions")
    a = ap.parse_args()

    if a.project:
        s = lsutil.ls_session(a.token)
        r = s.get(f"{a.url}/api/projects/{a.project}/export", params={"exportType": "JSON", "download_all_tasks": "false"})
        r.raise_for_status()
        tasks = r.json()
    elif a.export:
        tasks = json.load(open(a.export))
    else:
        ap.error("give --export FILE or --project ID")

    classes = lsutil.labels_from_config(a.label_config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = lsutil.MASK_KINDS + (("rectanglelabels",) if a.rects else ())
    n = 0
    for task in tasks:
        regions = list(lsutil.iter_regions(task, kinds))
        if not regions:
            continue
        W, H = regions[0][0]["original_width"], regions[0][0]["original_height"]
        sem = np.zeros((H, W), np.uint8)
        inst = np.zeros((H, W), np.uint16)
        inst_map = {}
        for i, (res, label) in enumerate(regions, 1):
            if label not in classes:
                classes.append(label)
            cid = classes.index(label) + 1
            m = lsutil.region_mask(res, W, H) > 0   # brush RLE / polygon / rectangle -> boolean mask
            sem[m] = cid
            inst[m] = i
            inst_map[i] = label
        stem = lsutil.image_stem(next(iter(task["data"].values())))
        if a.mode in ("semantic", "both"):
            cv2.imwrite(str(out / f"{stem}_semantic.png"), sem)
        if a.mode in ("instance", "both"):
            cv2.imwrite(str(out / f"{stem}_instance.png"), inst)
            (out / f"{stem}_instance.json").write_text(json.dumps(inst_map, indent=1))
        n += 1
    (out / "classes.json").write_text(json.dumps({"0": "background", **{str(i + 1): c for i, c in enumerate(classes)}}, indent=1))
    print(f"wrote masks for {n} images to {out}")


if __name__ == "__main__":
    main()
