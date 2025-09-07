from __future__ import annotations
import os
from pathlib import Path
from typing import List
from PIL import Image

from .settings import UPLOADS_DIR, RUNS_DIR

def stitch_recent_uploads(k: int = 3) -> Path:
    """Concatenate the most-recent k uploads horizontally → runs/stitched.jpg"""
    imgs: List[Path] = sorted(UPLOADS_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)[:k]
    if not imgs:
        raise FileNotFoundError("No images to stitch in /uploads.")

    pil = [Image.open(p).convert("RGB") for p in imgs]
    h = max(im.height for im in pil)
    pil = [im.resize((int(im.width * h / im.height), h)) if im.height != h else im for im in pil]
    total_w = sum(im.width for im in pil)
    stitched = Image.new("RGB", (total_w, h), (0, 0, 0))

    x = 0
    for im in pil:
        stitched.paste(im, (x, 0))
        x += im.width

    out_path = RUNS_DIR / "stitched.jpg"
    stitched.save(out_path, quality=95)
    return out_path
