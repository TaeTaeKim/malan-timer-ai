import cv2
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import logging
import easyocr
import numpy as np
from contextlib import asynccontextmanager
import time
import os
from extractor.stat_extractor import extract_all_stats_async

# Preload models on startup
MODEL_PATH = os.getenv("MODEL_PATH", "./best.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("malan_timer_ai")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Start Malan Timer AI FaseAPI application")
    global reader, model
    reader = easyocr.Reader(['en'], gpu=True)
    print(f"use model in {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logger.info("Pre-load YOLO and EasyOCR succeed")
    yield  # App runs here

app = FastAPI(lifespan=lifespan)

@app.post("/extract")
async def extract_stats(file: UploadFile = File(...)):
    # Read image data from uploade file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    request_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    start_time = time.time()
    result = extract_all_stats_async(request_img, model, reader)
    elapsed_time = time.time() - start_time
    logger.info(f"extract_stats_from_image executed in {elapsed_time:.4f} seconds")
    return {"extracted_data": result}