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
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    Display RAW (uploads) and YOLO-annotated (runs) images in Hogwarts style.
    Shows them side-by-side, sorted by obstacle number.
    """

    import re, os

    def extract_number(filename: str) -> int:
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 9999

    # Collect files
    raw_files = sorted(
        [f for f in os.listdir(UPLOADS_DIR) if f.endswith((".jpg", ".png"))],
        key=extract_number
    )
    run_files = []
    for root, _, files in os.walk(RUNS_DIR):
        for f in files:
            if f.endswith((".jpg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, f), RUNS_DIR)
                run_files.append(rel_path)
    run_files.sort(key=extract_number)

    pairs = list(zip(raw_files, run_files))

    # Build Hogwarts-themed HTML
    html = """
    <html>
    <head>
      <link href="https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&display=swap" rel="stylesheet">
      <style>
        body {
          background-image: url('/static/marauders-map.jpg');
          background-size: cover;
          font-family: 'Cinzel Decorative', cursive;
          color: #f8e8c0;
          text-shadow: 0 0 8px #a67c52;
        }
        h2 {
          text-align: center;
          font-size: 2.2em;
          margin-top: 20px;
        }
        .pair {
          display: flex;
          flex-direction: column;
          align-items: center;
          background: rgba(30,20,10,0.7);
          border: 2px solid gold;
          border-radius: 10px;
          margin: 15px;
          padding: 10px 20px;
          box-shadow: 0 0 15px #c19a6b;
          animation: shimmer 4s infinite;
        }
        .imgrow {
          display: flex;
          gap: 20px;
          justify-content: center;
        }
        img {
          width: 300px;
          border-radius: 8px;
          border: 2px solid gold;
          box-shadow: 0 0 12px gold;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        img:hover {
          transform: scale(1.05);
          box-shadow: 0 0 25px #ffd700;
        }
        @keyframes shimmer {
          0%   { box-shadow: 0 0 10px #b8860b; }
          50%  { box-shadow: 0 0 25px #ffd700; }
          100% { box-shadow: 0 0 10px #b8860b; }
        }
        p { margin: 4px; text-align: center; }
      </style>
    </head>
    <body>
      <h2>✨ Enchanted Image Recognition Results ✨</h2>
      <div style="display:flex;flex-wrap:wrap;justify-content:center;">
    """

    for idx, (raw, ann) in enumerate(pairs):
        obstacle_num = extract_number(raw)
        html += f"""
        <div class="pair">
          <p><b>Obstacle {obstacle_num if obstacle_num != 9999 else idx+1}</b></p>
          <div class="imgrow">
            <div>
              <img src="/uploads/{raw}">
              <p>RAW Image</p>
            </div>
            <div>
              <img src="/runs/{ann}">
              <p>Detected Result</p>
            </div>
          </div>
        </div>
        """

    html += """
      </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)