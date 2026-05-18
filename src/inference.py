"""
inference.py
------------
Production inference: load artefact, run predictions, write outputs.

Supports three modes (called from cli.py):
  score   — batch scoring of new monthly events
  predict — single-event check

Zone thresholds (data-driven from 14,073 labelled events):
  prob < 0.30  → green  / Keine Reklamation  (95.8% truly machine — reliable)
  0.30–0.69   → yellow / Unsicher            (genuinely ambiguous)
  prob ≥ 0.70  → red    / Reklamation        (62.7%+ truly paper)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import (
    extract_features, split_num_cat,
    get_valid_cameras, build_event_sequence, parse_metadata,
    load_new_events,
)
from .excel_export import write_coloured_excel

log = logging.getLogger(__name__)

DEFAULT_FOLD = 1


def load_artefact(model_dir: str, fold: int = DEFAULT_FOLD) -> dict:
    """
    Loads a saved fold artefact (model + scaler + OHE + threshold).

    Raises FileNotFoundError if the artefact doesn't exist.
    Raises RuntimeError if the artefact is missing scaler/OHE (old format).
    """
    path = Path(model_dir) / f"xgb_fold_{fold}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Artefact not found: {path}\n"
            "Run `python -m src.cli train` first."
        )
    with open(path, "rb") as f:
        art = pickle.load(f)

    if "scaler" not in art or "ohe" not in art:
        raise RuntimeError(
            f"Artefact at {path} is missing scaler/OHE. "
            "Re-run training to save complete artefacts."
        )

    log.info("Loaded fold %d  (AUC=%.4f, threshold=%.3f)",
             fold, art.get("auc", float("nan")), art["threshold"])
    return art


def _infer(X_seq: np.ndarray, event_data: list, art: dict) -> pd.DataFrame:
    """
    Core inference: feature extraction → preprocessing → predict → DataFrame.
    """
    raw     = extract_features(X_seq, event_data)
    X_num, X_cat = split_num_cat(raw)

    X_num_s = art["scaler"].transform(X_num)
    X_ohe   = art["ohe"].transform(X_cat)
    X       = np.hstack([X_num_s, X_ohe])

    probs   = art["model"].predict_proba(X)[:, 1]
    preds   = (probs >= art["threshold"]).astype(int)
    thr     = art["threshold"]

    return pd.DataFrame({
        "event_id":        [e["event_id"] for e in event_data],
        "source":          [e.get("source", "") for e in event_data],
        "prediction":      preds.tolist(),
        "prediction_str":  ["paper_problem" if p else "machine_problem" for p in preds],
        "probability":     probs.round(6),
        "score_%":         [round(p * 100, 2) for p in probs],
        "confidence_%":    [
            round(((p - thr) / (1 - thr) * 100) if p >= thr
                  else ((thr - p) / thr * 100), 2)
            for p in probs
        ],
        "confident_class": ["paper_problem" if p >= thr else "machine_problem" for p in probs],
        "einschaetzung":   [_zone_label(p) for p in probs],
        "farbe":           [_zone_colour(p) for p in probs],
        "threshold":       thr,
    })


def _zone_label(p: float) -> str:
    if p < 0.30:  return "Keine Reklamation"
    if p < 0.70:  return "Unsicher"
    return "Reklamation"


def _zone_colour(p: float) -> str:
    if p < 0.30:  return "grün"
    if p < 0.70:  return "gelb"
    return "rot"


def score(model_dir: str, input_dir: str, output_dir: str,
          fold: int = DEFAULT_FOLD) -> None:
    """
    Scores all events in input_dir and writes labelled CSV + colour Excel.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    art = load_artefact(model_dir, fold=fold)
    X_seq, event_data, skipped = load_new_events(input_dir)

    log.info("Running inference on %d events (fold %d)...", len(event_data), fold)
    results_df = _infer(X_seq, event_data, art)
    results_df["fold_used"] = fold
    results_df["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv  = Path(output_dir) / f"predictions_{ts}.csv"
    out_xlsx = Path(output_dir) / f"predictions_{ts}.xlsx"

    results_df.to_csv(out_csv, index=False)
    write_coloured_excel(results_df, str(out_xlsx))

    _print_score_summary(results_df, skipped, out_csv, out_xlsx)


def predict_single(json_path: str, art: dict) -> dict:
    """
    Scores a single event JSON file.

    Returns:
        dict with event_id, prediction, label_str, probability, threshold
    """
    import json
    with open(json_path, encoding="utf-8") as f:
        ev = json.load(f)

    valid_cams = get_valid_cameras(ev.get("videos", []))
    if not valid_cams:
        raise ValueError("No valid cameras in event.")

    X_seq_ev, cam_meta = build_event_sequence(valid_cams)
    if X_seq_ev is None:
        raise ValueError("Could not build event sequence.")

    event_id = Path(json_path).name.replace("_event.json", "")
    ev_data  = [{"event_id": event_id, "source": Path(json_path).name,
                 "cam_meta": cam_meta, "metadata": parse_metadata(ev)}]
    X_seq_1  = X_seq_ev[np.newaxis, ...]

    df  = _infer(X_seq_1, ev_data, art)
    row = df.iloc[0]

    return {
        "event_id":    event_id,
        "prediction":  int(row["prediction"]),
        "label_str":   row["prediction_str"],
        "probability": float(row["probability"]),
        "score_%":     float(row["score_%"]),
        "einschaetzung": row["einschaetzung"],
        "farbe":       row["farbe"],
        "threshold":   art["threshold"],
    }


def _print_score_summary(df, skipped, out_csv, out_xlsx):
    n_green  = (df["farbe"] == "grün").sum()
    n_yellow = (df["farbe"] == "gelb").sum()
    n_red    = (df["farbe"] == "rot").sum()

    log.info("=" * 60)
    log.info("SCORE COMPLETE")
    log.info("  Events scored         : %d", len(df))
    log.info("  Keine Reklamation (green) : %d  (prob < 0.30)", n_green)
    log.info("  Unsicher          (yellow): %d  (0.30 <= prob < 0.70)", n_yellow)
    log.info("  Reklamation       (red)   : %d  (prob >= 0.70)", n_red)
    log.info("  Skipped                   : %d", len(skipped))
    log.info("  Output CSV   : %s", out_csv)
    log.info("  Output Excel : %s", out_xlsx)
    log.info("=" * 60)

    if skipped:
        log.warning("Skipped events:")
        for s in skipped:
            log.warning("  • %s", s)
