"""
api.py — FastAPI Serving Layer for Clothing Fit Prediction
===========================================================
Exposes the trained fit prediction model as a REST API.

Endpoints:
    POST /predict  — predict fit (Small/Fit/Large) for a single item
    POST /batch    — predict fit for multiple items
    GET  /health   — health check
    GET  /info     — model info and feature list

Usage:
    uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import PredictRequest, PredictResponse, BatchRequest, BatchResponse
from src.model import FitModel
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clothing Fit Prediction API",
    description="Predicts whether clothing items run Small, Fit, or Large",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model on startup
model = FitModel()

@app.on_event("startup")
async def startup():
    logger.info("Loading fit prediction model...")
    model.load("models/fit_predictor_nn.pt")
    logger.info("Model loaded ✅")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model.is_loaded}

@app.get("/info")
def info():
    return {
        "model": "PyTorch FeedForward Neural Network",
        "classes": ["small", "fit", "large"],
        "features": model.feature_cols if model.is_loaded else [],
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        result = model.predict(request)
        result.latency_ms = round((time.time() - start) * 1000, 2)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=BatchResponse)
def batch_predict(request: BatchRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        results = [model.predict(r) for r in request.items]
        return BatchResponse(
            predictions=results,
            count=len(results),
            latency_ms=round((time.time() - start) * 1000, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
