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
    Save YOLO annotated image + add a compact legend box in the true top-right corner.
    Shows label + id only (no confidence).
    """
    out_parent.mkdir(parents=True, exist_ok=True)
    out_file = out_parent / "image0.jpg"

    # Base image (YOLO annotated with its magenta tags)
    img = np.squeeze(results.render()[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    df = results.pandas().xyxy[0]
    h, w, _ = img.shape

    # ---- Collect all detection labels ----
    lines = []
    for _, row in df.iterrows():
        cls = str(row["name"])
        label, image_id = LEGEND.get(cls, (cls, None))
        if image_id is not None:
            lines.append(f"{label} (id={image_id})")
        else:
            lines.append(label)

    if not lines:
        cv2.imwrite(str(out_file), img)
        return out_file

    # ---- Calculate box size ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    margin = 5
    line_height = 18

    text_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = max(text_width, tw)

    box_w = text_width + margin * 2
    box_h = line_height * len(lines) + margin * 2

    # ---- Top-right position ----
    x0, y0 = w - box_w - 5, 5
    x1, y1 = w - 5, 5 + box_h

    # ---- Draw white rectangle ----
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

    # ---- Write text lines ----
    y_text = y0 + margin + 12
    for line in lines:
        cv2.putText(img, line, (x0 + margin, y_text),
                    font, font_scale, (0, 0, 0), thickness)
        y_text += line_height

    cv2.imwrite(str(out_file), img)
    return out_file