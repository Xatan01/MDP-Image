from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import os
from datetime import datetime  
import re

from server.settings import (SRV_DIR, RUNS_DIR, UPLOADS_DIR, IMG_SIZE,
                             CONF_THRESH, IOU_THRESH, DEVICE, WEIGHTS_PATH)
from server.utils import read_pil
from server.yolo_service import run_inference, save_annotated
from server.heuristics import pick_label_from_detections
from server.stitcher import stitch_recent_uploads

app = FastAPI(title="MDP YOLOv5 Inference Server", version="1.0.0")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


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
    # read uploaded image
    pil_img: Image.Image = read_pil(file)

    # ✅ Save using the original uploaded filename (e.g. capture_1.jpg)
    upload_path = UPLOADS_DIR / file.filename
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    pil_img.save(upload_path, quality=95)

    # run YOLO
    df, results = run_inference(pil_img)

    # pick largest bounding box (after YOLO confidence/IOU filtering)
    if df is not None and not df.empty:
        df["area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
        best = df.loc[df["area"].idxmax()]
        label = str(best["name"])

        if label.isdigit():
            number = int(label)
        elif label.lower() == "marker":
            number = -1
        else:
            number = -1
    else:
        number = -1

    # ✅ Save annotated image directly to /runs with the same name (no exp folders)
    save_annotated(results, RUNS_DIR / file.filename)

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
    """
    Display RAW (from uploads) and YOLO-annotated (from runs) images side-by-side.
    Sorts by obstacle number extracted from filename (e.g., capture_1.jpg → 1).
    """

    def extract_number(filename: str) -> int:
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 9999  # large fallback for non-numbered files

    # --- Collect RAW images ---
    raw_files = sorted(
        [f for f in os.listdir(UPLOADS_DIR) if f.endswith((".jpg", ".png"))],
        key=extract_number
    )

    # --- Collect YOLO-annotated images ---
    run_files = []
    for root, _, files in os.walk(RUNS_DIR):
        for f in files:
            if f.endswith((".jpg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, f), RUNS_DIR)
                run_files.append(rel_path)
    run_files.sort(key=extract_number)

    # --- Pair them ---
    pairs = list(zip(raw_files, run_files))

    # --- Build HTML ---
    html = """
    <html>
    <body style="background-color:#e6f2ff; font-family:Arial, sans-serif;">
    <h2>Task 1 — Image Recognition Results</h2>
    <p>Each row shows the RAW camera image (left) and the YOLO-detected result (right).<br>
    Sorted by obstacle number from filename.</p>
    <div style="display:flex;flex-wrap:wrap;gap:10px;">
    """

    for idx, (raw, ann) in enumerate(pairs):
        obstacle_num = extract_number(raw)
        html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin:10px;
                    background:#fff;padding:8px;border-radius:8px;
                    box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <p><b>Obstacle {obstacle_num if obstacle_num != 9999 else idx+1}</b></p>
            <div style="display:flex;gap:8px;align-items:center;">
                <div>
                    <img src="/uploads/{raw}" width="300" style="border:1px solid #ccc;">
                    <p style="text-align:center;margin:4px 0;">RAW Image</p>
                </div>
                <div>
                    <img src="/runs/{ann}" width="300" style="border:2px solid #000;">
                    <p style="text-align:center;margin:4px 0;">Annotated Result</p>
                </div>
            </div>
        </div>
        """

    html += "</div></body></html>"
    return HTMLResponse(content=html)