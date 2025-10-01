from __future__ import annotations
from . import paths_fix  # ensure hot-fix executes early

import os
from pathlib import Path

# Base dirs
SRV_DIR: Path = Path(__file__).resolve().parent   # .../YOLOv5
ROOT:    Path = SRV_DIR.parent                           # .../MDP

# Runtime config (env or defaults)
IMG_SIZE    = int(os.environ.get("IMG_SIZE", "640"))
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.35"))
IOU_THRESH  = float(os.environ.get("IOU_THRESH", "0.45"))
DEVICE      = os.environ.get("DEVICE", "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != "nt" else "cpu")
HUB_FORCE_RELOAD = os.environ.get("HUB_FORCE_RELOAD", "1") not in ("0", "false", "False", "")

# Folders
UPLOADS_DIR = SRV_DIR / "uploads"
RUNS_DIR    = SRV_DIR / "runs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def resolve_weights() -> Path:
    env_path = os.environ.get("WEIGHTS_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
        raise RuntimeError(f"WEIGHTS_PATH set but not found: {p}")

    for c in [SRV_DIR / "weights" / "best.pt", ROOT / "Weights" / "best.pt", ROOT / "weights" / "best.pt", ROOT.parent / "Weights" / "best.pt"]:
        if c.exists():
            return c.resolve()

    # last resort: first .pt anywhere under repo
    found = list(ROOT.rglob("*.pt"))
    if found:
        return found[0].resolve()

    raise RuntimeError("No weights found. Place 'best.pt' in 'YOLOv5/weights' or set WEIGHTS_PATH.")

WEIGHTS_PATH = resolve_weights()
