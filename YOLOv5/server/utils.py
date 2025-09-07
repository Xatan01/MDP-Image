from __future__ import annotations
import io, uuid
from datetime import datetime
from typing import Any
from PIL import Image
from fastapi import UploadFile, HTTPException

def code_to_label(code: str) -> str:
    """11..19→'1'..'9', 20..40→'A'..'U', 'marker' stays 'marker'."""
    if code == "marker":
        return "marker"
    try:
        n = int(code)
        if 11 <= n <= 19:
            return str(n - 10)
        if 20 <= n <= 40:
            return chr(ord('A') + (n - 20))
    except Exception:
        pass
    return code

def read_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    return Image.open(io.BytesIO(data)).convert("RGB")

def unique_stem() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
