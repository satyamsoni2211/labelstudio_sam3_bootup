"""Label Studio ML backend: SAM3 (Sam3Tracker, SAM2-style PVS) with click / box prompts.

Interactive flow: LS sends the *context* (keypoints + rectangle drawn so far). We run SAM3 and
return a mask whose label is the label of the prompt:
  SAM3_OUTPUT=brush    (default) one BrushLabels region, RLE like the official SAM2 example
  SAM3_OUTPUT=polygon  one PolygonLabels region (largest connected component, % coordinates)
  SAM3_OUTPUT=both     both regions (the user then has two overlapping regions to delete from)
"""
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_tools.core.utils.io import get_local_path
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
EMBED_CACHE_SIZE = int(os.getenv("SAM3_EMBED_CACHE", "6"))
# Polygon simplification: epsilon = SIMPLIFY * contour perimeter (smaller = more points)
SIMPLIFY = float(os.getenv("SAM3_POLY_EPSILON", "0.0015"))
MAX_POINTS = int(os.getenv("SAM3_POLY_MAX_POINTS", "400"))
OUTPUT = os.getenv("SAM3_OUTPUT", "brush").strip().lower()
if OUTPUT not in ("brush", "polygon", "both"):
    log.warning("SAM3_OUTPUT=%r invalid, using 'brush'", OUTPUT)
    OUTPUT = "brush"


def pick_device() -> str:
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()
_model = None
_processor = None
_embed_cache: "OrderedDict[str, tuple]" = OrderedDict()


def load_model():
    """Load SAM3 once per process."""
    global _model, _processor
    if _model is None:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        t0 = time.time()
        log.info("Loading %s on %s ...", MODEL_ID, DEVICE)
        _processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
        _model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        log.info("SAM3 loaded in %.1fs; model device=%s dtype=%s",
                 time.time() - t0, next(_model.parameters()).device, next(_model.parameters()).dtype)
    return _model, _processor


def mask_to_polygon(mask: np.ndarray) -> Optional[List[List[float]]]:
    """Binary HxW mask -> simplified polygon [[x%, y%], ...] of the largest component."""
    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = SIMPLIFY * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > MAX_POINTS:
        eps *= 1.5
        approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return None
    return [[round(x / w * 100, 4), round(y / h * 100, 4)] for x, y in pts]


class SAM3Backend(LabelStudioMLBase):
    # NOTE: PyPI `label-studio-ml` 1.0.x API: predict() returns a plain list of prediction dicts
    # and the label config is available as self.parsed_label_config.
    MODEL_VERSION = f"sam3-tracker-{DEVICE}"

    def _output_tags(self):
        """{'brush': (from_name, to_name, value_key), 'polygon': ...} for the requested SAM3_OUTPUT."""
        wanted = {"brush": ["brush"], "polygon": ["polygon"], "both": ["brush", "polygon"]}[OUTPUT]
        types = {"brush": "BrushLabels", "polygon": "PolygonLabels"}
        found = {}
        for kind in wanted:
            for from_name, info in self.parsed_label_config.items():
                if info["type"] == types[kind]:
                    found[kind] = (from_name, info["to_name"][0], info["inputs"][0]["value"])
                    break
        if not found:
            raise ValueError(f"SAM3_OUTPUT={OUTPUT} but the label config has no "
                             f"{' / '.join(types[k] for k in wanted)} tag bound to an Image")
        return found

    # -- image + embeddings -------------------------------------------------
    def _load_image(self, task: Dict, value: str) -> (str, Image.Image):
        # Always download through the LS API (the container's /data/upload is not on this Mac's disk;
        # image_dir -> nonexistent path disables the "look in the local LS media dir" shortcut).
        path = get_local_path(
            task["data"][value],
            image_dir="/nonexistent",
            hostname=os.getenv("LABEL_STUDIO_URL") or self.hostname,
            access_token=os.getenv("LABEL_STUDIO_API_KEY") or self.access_token,
            task_id=task.get("id"),
        )
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # LS displays EXIF-rotated images
        return path, img

    @torch.no_grad()
    def _segment(self, key: str, image: Image.Image, points, labels, box):
        model, processor = load_model()
        kw = {}
        if points:
            kw["input_points"] = [[points]]      # [image][object][point][xy]
            kw["input_labels"] = [[labels]]      # [image][object][point]
        if box is not None:
            kw["input_boxes"] = [[box]]          # [image][box][xyxy]
        t0 = time.time()
        cached = _embed_cache.get(key)
        if cached is None:
            # cold: preprocess the image (resize/normalise) and run the vision encoder once
            inputs = processor(images=image, return_tensors="pt", **kw)
            inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
            embeds = model.get_image_embeddings(inputs.pop("pixel_values"))
            _embed_cache[key] = (embeds, [image.size[1], image.size[0]])
            while len(_embed_cache) > EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
            cache_hit = False
        else:
            # warm: skip image preprocessing entirely, only normalise the prompt coordinates
            _embed_cache.move_to_end(key)
            embeds, hw = cached
            inputs = processor(original_sizes=[hw], return_tensors="pt", **kw)
            inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
            cache_hit = True
        original_sizes = inputs["original_sizes"]
        fwd = {k: v for k, v in inputs.items() if k != "original_sizes" and k != "reshaped_input_sizes"}
        single_click = bool(points) and len(points) == 1 and box is None
        out = model(image_embeddings=embeds, multimask_output=single_click, **fwd)
        masks = processor.post_process_masks(out.pred_masks.cpu(), original_sizes.cpu())[0]  # [obj, n, H, W]
        scores = out.iou_scores.float().cpu().reshape(-1)
        masks = masks.reshape(-1, *masks.shape[-2:])
        best = int(torch.argmax(scores))
        if DEVICE == "mps":
            torch.mps.synchronize()
        log.info("SAM3 inference %.0f ms (embedding cache %s), best iou=%.3f",
                 (time.time() - t0) * 1000, "hit" if cache_hit else "miss", float(scores[best]))
        return masks[best].numpy().astype(np.uint8), float(scores[best])

    # -- LS API -------------------------------------------------------------
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        if not context or not context.get("result"):
            # LS "Test Request" / background pre-annotation: there is nothing to segment without a click/box.
            log.info("predict called without interactive context (e.g. 'Test Request' in LS): returning [] - "
                     "this is normal. Turn on Auto-Annotation and click/drag on an image to get a mask.")
            return []

        tags = self._output_tags()
        value = next(iter(tags.values()))[2]
        task = tasks[0]
        path, image = self._load_image(task, value)
        W, H = image.size

        points, plabels, box, label = [], [], None, None
        for ctx in context["result"]:
            v, t = ctx["value"], ctx["type"]
            x, y = v["x"] * W / 100.0, v["y"] * H / 100.0
            label = (v.get(t) or [label])[0]
            if t == "keypointlabels":
                points.append([x, y])
                plabels.append(int(ctx.get("is_positive", 1)))
            elif t == "rectanglelabels":
                box = [x, y, x + v["width"] * W / 100.0, y + v["height"] * H / 100.0]
        if not points and box is None:
            return []

        key = f"{path}:{os.path.getmtime(path)}"
        mask, score = self._segment(key, image, points, plabels, box)
        results = []
        common = {"original_width": W, "original_height": H, "image_rotation": 0, "score": score, "readonly": False}
        if "brush" in tags:
            from label_studio_converter.brush import mask2rle
            from_name, to_name, _ = tags["brush"]
            results.append({
                "id": uuid.uuid4().hex[:10], "from_name": from_name, "to_name": to_name, "type": "brushlabels",
                "value": {"format": "rle", "rle": mask2rle(mask.astype(np.uint8) * 255), "brushlabels": [label]},
                **common})
        if "polygon" in tags:
            poly = mask_to_polygon(mask)
            if poly is not None:
                from_name, to_name, _ = tags["polygon"]
                results.append({
                    "id": uuid.uuid4().hex[:10], "from_name": from_name, "to_name": to_name, "type": "polygonlabels",
                    "value": {"points": poly, "closed": True, "polygonlabels": [label]}, **common})
        if not results:
            return []
        log.info("interactive predict: %d point(s), box=%s, label=%s -> %s, mask %d px, score %.3f",
                 len(points), "yes" if box else "no", label, "+".join(r["type"] for r in results),
                 int(mask.sum()), score)
        return [{"result": results, "score": score, "model_version": self.MODEL_VERSION}]

    def fit(self, tasks, workdir=None, **kwargs):
        # LS sends annotation events (webhook / train) when annotations are submitted or updated; SAM3 is not trained.
        return {}

    def process_event(self, event, data, job_id, additional_params):
        log.info("ignoring LS event %s (no training)", event)
        return {}
