from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import os
from datetime import datetime  # ✅ added for unique filenames

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
    # save raw upload with unique timestamp name
    pil_img: Image.Image = read_pil(file)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    upload_path = UPLOADS_DIR / f"upload_{ts}.jpg"
    pil_img.save(upload_path, quality=95)

    # run YOLO
    df, results = run_inference(pil_img)

    # pick largest bounding box (after YOLO confidence/IOU filtering)
    if df is not None and not df.empty:
        df["area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
        best = df.loc[df["area"].idxmax()]
        label = str(best["name"])  # always a string

        if label.isdigit():
            number = int(label)  # numeric class names
        elif label.lower() == "marker":
            number = "BULLSEYE"  # special case
        else:
            number = -1          # unknown / unsupported
    else:
        number = -1

    # save annotated preview locally
    save_annotated(results, RUNS_DIR / upload_path.stem[:8])

    # final API response
    return JSONResponse({"target": number, "obstacle_id": 1})


@app.post("/stitch")
def stitch() -> JSONResponse:
    try:
        out = stitch_recent_uploads(k=3)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"stitched_path": f"/runs/{out.name}", "count": 3})


@app.get("/gallery", response_class=HTMLResponse)
def gallery():
    # look inside RUNS_DIR for saved images
    image_urls = []
    for root, _, files in os.walk(RUNS_DIR):
        for file in files:
            if file.endswith((".jpg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, file), RUNS_DIR)
                image_urls.append(f"/runs/{rel_path}")

    # build HTML grid
    html_content = "<html><body><h2>Inference Results</h2><div style='display:flex;flex-wrap:wrap;'>"
    for url in image_urls:
        html_content += f"<div style='margin:5px;'><img src='{url}' width='300'></div>"
    html_content += "</div></body></html>"

    return HTMLResponse(content=html_content)