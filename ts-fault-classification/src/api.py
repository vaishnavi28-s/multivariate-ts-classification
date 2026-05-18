"""
api.py
------
FastAPI microservice for single-event and batch prediction.

Endpoints:
  GET  /health          — liveness check
  POST /predict         — single event prediction
  POST /predict/batch   — batch prediction (list of events)

The model artefact is loaded once at startup from the configured
model directory and fold. Set via environment variables:
  MODEL_DIR    (default: ./processed_data/xgb_models)
  DEFAULT_FOLD (default: 1)
"""

import os
import json
import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .preprocessing import (
    get_valid_cameras, build_event_sequence, parse_metadata,
    extract_features, split_num_cat,
)
from .inference import load_artefact, _infer, _zone_label, _zone_colour

log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ts-fault-classification",
    description="Industrial fault classification from camera CV scores + metadata.",
    version="1.0.0",
)

MODEL_DIR    = os.getenv("MODEL_DIR", "./processed_data/xgb_models")
DEFAULT_FOLD = int(os.getenv("DEFAULT_FOLD", "1"))

_artefact = None


@app.on_event("startup")
def _load_model():
    global _artefact
    try:
        _artefact = load_artefact(MODEL_DIR, fold=DEFAULT_FOLD)
        log.info("Model loaded (fold %d, AUC=%.4f)", DEFAULT_FOLD, _artefact.get("auc", 0))
    except Exception as e:
        log.error("Failed to load model: %s", e)


# ── Request / response schemas ────────────────────────────────────────────────

class FrameScore(BaseModel):
    no_defect:    float
    defect:       float
    rollenwechsel: float
    Kantenfehler: float
    tear:         float


class Frame(BaseModel):
    name:   str
    label:  str = ""
    scores: FrameScore


class Camera(BaseModel):
    camera_id:   str
    camera_name: str = ""
    frames:      list[Frame]


class EventRequest(BaseModel):
    printer:        str | None = None
    speed:          Any        = None
    grammage_weight: Any       = None
    web_width:      Any        = None
    pap_len:        Any        = None
    detector:       Any        = None
    grade:          str | None = None
    paper_supplier: str | None = None
    date_time_str:  str | None = None
    videos:         list[Camera]


class PredictionResponse(BaseModel):
    prediction:    int
    label:         str
    probability:   float
    score_pct:     float
    einschaetzung: str
    farbe:         str
    threshold:     float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _artefact is not None,
        "fold": DEFAULT_FOLD,
        "auc": round(_artefact.get("auc", 0), 4) if _artefact else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(event: EventRequest):
    if _artefact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ev_dict = event.model_dump()
    ev_dict["videos"] = [
        {"camera_id": c["camera_id"], "camera_name": c["camera_name"],
         "frames": [{"name": f["name"], "label": f["label"],
                     "scores": f["scores"]} for f in c["frames"]]}
        for c in ev_dict["videos"]
    ]

    valid_cams = get_valid_cameras(ev_dict["videos"])
    if not valid_cams:
        raise HTTPException(status_code=422, detail="No valid camera frames found")

    X_seq_ev, cam_meta = build_event_sequence(valid_cams)
    if X_seq_ev is None:
        raise HTTPException(status_code=422, detail="Could not build event sequence")

    ev_data = [{"event_id": "api_request", "source": "api",
                "cam_meta": cam_meta, "metadata": parse_metadata(ev_dict)}]
    X       = X_seq_ev[np.newaxis, ...]
    df      = _infer(X, ev_data, _artefact)
    row     = df.iloc[0]

    return PredictionResponse(
        prediction=int(row["prediction"]),
        label=row["prediction_str"],
        probability=float(row["probability"]),
        score_pct=float(row["score_%"]),
        einschaetzung=row["einschaetzung"],
        farbe=row["farbe"],
        threshold=float(row["threshold"]),
    )


@app.post("/predict/batch")
def predict_batch(events: list[EventRequest]):
    if _artefact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [predict(ev) for ev in events]
