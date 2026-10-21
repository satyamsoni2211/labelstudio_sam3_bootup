"""Shared helpers: .env loading, LS API session, label-config parsing, annotation iteration."""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env"):
    """Minimal .env loader (does not override already-set environment variables)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY", "")


def ls_session(token: str = None) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Token {token or LS_TOKEN}"
    return s


def labels_from_config(xml_path) -> list:
    """Ordered label names of the BrushLabels / PolygonLabels tag (class id = index)."""
    root = ET.parse(xml_path).getroot()
    tag = None
    for name in ("BrushLabels", "PolygonLabels", "RectangleLabels"):
        tag = root.find(f".//{name}")
        if tag is not None:
            break
    return [l.get("value") for l in tag.findall("Label")] if tag is not None else []


_PREFIX = re.compile(r"^[0-9a-f]{8}-")


def image_stem(image_ref: str) -> str:
    """LS prefixes uploaded files with an 8-hex-char hash ('ab12cd34-cat.jpg'); strip it."""
    name = Path(image_ref.split("?")[0]).name
    stem = Path(name).stem
    return _PREFIX.sub("", stem)


MASK_KINDS = ("polygonlabels", "brushlabels")   # regions that carry a segmentation mask


def iter_regions(task, kinds=MASK_KINDS):
    """Yield (result_dict, label) for the first non-cancelled annotation of a task (JSON export format)."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return
    for r in anns[-1]["result"]:  # latest annotation
        if r.get("type") in kinds and r["value"].get(r["type"]):
            yield r, r["value"][r["type"]][0]


def poly_px(value, W, H):
    return [(x * W / 100.0, y * H / 100.0) for x, y in value["points"]]


def rect_px(value, W, H):
    x, y = value["x"] * W / 100.0, value["y"] * H / 100.0
    return x, y, x + value["width"] * W / 100.0, y + value["height"] * H / 100.0


def region_mask(res, W, H):
    """uint8 HxW (0/1) mask of a brushlabels (RLE), polygonlabels or rectanglelabels region."""
    t, v = res["type"], res["value"]
    if t == "brushlabels":
        from label_studio_converter.brush import decode_rle
        w, h = res.get("original_width", W), res.get("original_height", H)
        m = (np.reshape(decode_rle(v["rle"]), (h, w, 4))[:, :, 3] > 0).astype(np.uint8)
        if (w, h) != (W, H):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m
    m = np.zeros((H, W), np.uint8)
    if t == "polygonlabels":
        cv2.fillPoly(m, [np.round(np.array(poly_px(v, W, H))).astype(np.int32)], 1)
    else:
        x0, y0, x1, y1 = [int(round(c)) for c in rect_px(v, W, H)]
        m[y0:y1, x0:x1] = 1
    return m


def mask_polygons(mask, epsilon=0.0015, min_area=4):
    """Binary mask -> list of Nx2 float arrays (pixel coords), one per external contour (largest first)."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        a = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
        if len(a) >= 3:
            out.append(a)
    return out


def region_geometry(res, W, H):
    """(polygons, bbox, area): polygons = list of Nx2 pixel arrays (none for rectangles), bbox = x0,y0,x1,y1, area px."""
    t, v = res["type"], res["value"]
    if t == "rectanglelabels":
        x0, y0, x1, y1 = rect_px(v, W, H)
        return [], (x0, y0, x1, y1), (x1 - x0) * (y1 - y0)
    if t == "polygonlabels":
        pts = np.array(poly_px(v, W, H))
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return [pts], (x.min(), y.min(), x.max(), y.max()), area
    m = region_mask(res, W, H)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return [], (0, 0, 0, 0), 0
    return mask_polygons(m), (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), int(m.sum())
