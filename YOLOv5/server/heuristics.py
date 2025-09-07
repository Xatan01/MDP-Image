from __future__ import annotations
from typing import Any, Dict, List
from .utils import code_to_label

def pick_label_from_detections(df) -> str | None:
    """
    df columns: xmin, ymin, xmax, ymax, confidence, class, name
    Rule:
      1) choose largest area **excluding 'marker'**
      2) if none left, choose highest-confidence among all
    """
    dets: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        x1, y1, x2, y2 = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        dets.append({"area": area, "conf": float(r["confidence"]), "code": str(r["name"])})

    if not dets:
        return None

    non_marker = [d for d in dets if d["code"] != "marker"]
    chosen = max(non_marker, key=lambda d: d["area"]) if non_marker else max(dets, key=lambda d: d["conf"])
    return code_to_label(chosen["code"])
