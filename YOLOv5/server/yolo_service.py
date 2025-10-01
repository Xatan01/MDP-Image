from __future__ import annotations
import threading
from pathlib import Path
from typing import Any, Tuple
import numpy as np
import cv2
from .legend import LEGEND

import torch
from PIL import Image

from .settings import (WEIGHTS_PATH, IMG_SIZE, CONF_THRESH, IOU_THRESH, DEVICE,
                       HUB_FORCE_RELOAD, SRV_DIR, RUNS_DIR)
from .utils import unique_stem

model_lock = threading.Lock()

def _load_model() -> Any:
    """Load YOLOv5 model via local repo (if present) or torch.hub."""
    local_repo = SRV_DIR / "yolov5"
    if local_repo.exists():
        mdl = torch.hub.load(str(local_repo), 'custom', path=str(WEIGHTS_PATH), source='local', trust_repo=True)
    else:
        mdl = torch.hub.load('ultralytics/yolov5', 'custom', path=str(WEIGHTS_PATH),
                             source='github', trust_repo=True, force_reload=HUB_FORCE_RELOAD)
    mdl = mdl.to(DEVICE)
    mdl.conf = CONF_THRESH
    mdl.iou  = IOU_THRESH
    try:
        mdl.autoshape()
    except Exception:
        pass
    return mdl

MODEL = _load_model()

def run_inference(pil_img: Image.Image):
    with model_lock:
        results = MODEL(pil_img, size=IMG_SIZE)

    df = results.pandas().xyxy[0]
    df = df.rename(columns={"class": "target_id"})   # ✅ only rename, don’t drop xmin,ymin,etc

    return df, results

def save_annotated(results, out_parent: Path) -> Path:
    """
    Save YOLO annotated image + add one compact legend box in the top-right corner.
    Shows exactly two lines: label + id (light font, compact box).
    """
    out_parent.mkdir(parents=True, exist_ok=True)
    out_file = out_parent / "image0.jpg"

    # Render YOLO detections (boxes only)
    img = np.squeeze(results.render()[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    df = results.pandas().xyxy[0]
    h, w, _ = img.shape

    # Pick the most confident detection
    if df is None or df.empty:
        cv2.imwrite(str(out_file), img)
        return out_file

    best = df.loc[df["confidence"].idxmax()]
    cls = str(best["name"])
    label, image_id = LEGEND.get(cls, (cls, None))

    if image_id is None:
        cv2.imwrite(str(out_file), img)
        return out_file

    # Two lines of text
    text_top = f"{label}"
    text_bottom = f"id={image_id}"

    # Font settings (lighter + smaller)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5   # smaller
    thickness = 1      # thinner
    margin = 4
    line_height = 16   # tighter spacing

    # Get text sizes
    (w1, h1), _ = cv2.getTextSize(text_top, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(text_bottom, font, font_scale, thickness)

    box_w = max(w1, w2) + margin * 2
    box_h = h1 + h2 + line_height

    # Position box in top-right corner
    x0, y0 = w - box_w - 10, 10
    x1, y1 = x0 + box_w, y0 + box_h

    # Draw filled white rectangle + border
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

    # Draw text lines
    y_text = y0 + margin + h1
    cv2.putText(img, text_top, (x0 + margin, y_text),
                font, font_scale, (0, 0, 0), thickness)
    cv2.putText(img, text_bottom, (x0 + margin, y_text + line_height),
                font, font_scale, (0, 0, 0), thickness)

    cv2.imwrite(str(out_file), img)
    return out_file