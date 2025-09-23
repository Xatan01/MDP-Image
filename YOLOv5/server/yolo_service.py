from __future__ import annotations
import threading
from pathlib import Path
from typing import Any, Tuple

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
    """Save annotated image directly into runs/ without creating timestamp folders."""
    out_parent.mkdir(parents=True, exist_ok=True)
    out_file = out_parent / "image0.jpg"
    with model_lock:
        results.save(save_dir=str(out_parent))
    # rename/move YOLO's default save to overwrite consistently
    saved = list(out_parent.glob("*.jpg"))
    if saved:
        saved[0].rename(out_file)
        return out_file
    return out_parent