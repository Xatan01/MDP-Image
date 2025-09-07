from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server.settings import (SRV_DIR, RUNS_DIR, UPLOADS_DIR, IMG_SIZE,
                             CONF_THRESH, IOU_THRESH, DEVICE, WEIGHTS_PATH)
from server.utils import read_pil
from server.yolo_service import run_inference, save_annotated
from server.heuristics import pick_label_from_detections
from server.stitcher import stitch_recent_uploads

app = FastAPI(title="MDP YOLOv5 Inference Server", version="1.0.0")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "weights": str(WEIGHTS_PATH),
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "conf": CONF_THRESH,
        "iou": IOU_THRESH,
    }

@app.post("/image")
def infer_image(file: UploadFile = File(...)) -> JSONResponse:
    # save raw upload
    pil_img: Image.Image = read_pil(file)
    upload_path = UPLOADS_DIR / f"{file.filename or 'upload'}.jpg"
    pil_img.save(upload_path, quality=95)

    # run YOLO
    df, results = run_inference(pil_img)

    # choose final label using repo-like heuristic
    label = pick_label_from_detections(df) or "unknown"

    # save annotated preview
    saved = save_annotated(results, RUNS_DIR / upload_path.stem[:8])  # subfolder by day prefix
    # (client in repo doesn’t need annotated URL, but nice to have)
    _ = saved  # not returned to keep payload minimal

    # repo-shaped payload
    return JSONResponse({"image_id": label, "obstacle_id": 1})

@app.post("/stitch")
def stitch() -> JSONResponse:
    try:
        out = stitch_recent_uploads(k=3)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"stitched_path": f"/runs/{out.name}", "count": 3})
