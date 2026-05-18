"""
preprocessing.py
----------------
Event loading, sequence building, and feature extraction.

Handles:
  - Validating and parsing event JSON files (loose or inside ZIPs)
  - Extracting 300-frame camera windows aligned to the tear-score peak
  - Building flat feature vectors (68 numeric + 3 categorical)
  - Parsing metadata fields with unit stripping

Feature vector layout (fixed — never reorder):
  [0:58]  camera features  (29 per camera × 2 cameras)
  [58:60] cross-camera tear diff
  [60:63] cam meta (padding fractions, camera2 present flag)
  [63:68] numeric metadata (speed, grammage, web_width, pap_len, detector)
  [68:71] categorical metadata (printer, grade, paper_supplier) — raw, OHE applied later
"""

import re
import json
import zipfile
import logging
from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TIMESTEPS            = 300
NUM_FEATURES_PER_CAM = 5
MIN_VALID_FRAMES     = 10
SCORE_KEYS           = ["no_defect", "defect", "rollenwechsel", "Kantenfehler", "tear"]

# Feature name registry (68 numeric — fixed order)
_SCORE_NAMES = ["no_defect", "defect", "rollenwechsel", "kantenfehler", "tear"]
_CAM_NAMES   = ["cam1", "cam2"]

NUM_FEATURE_NAMES: list[str] = []
for _cam in _CAM_NAMES:
    for _stat in ["mean", "std", "max"]:
        for _s in _SCORE_NAMES:
            NUM_FEATURE_NAMES.append(f"{_cam}_{_s}_{_stat}")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_slope")
    NUM_FEATURE_NAMES.append(f"{_cam}_no_defect_slope")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_mean_last50")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_mean_first50")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_diff_50")
    NUM_FEATURE_NAMES.append(f"{_cam}_entropy_mean")
    NUM_FEATURE_NAMES.append(f"{_cam}_entropy_std")
    for _s in _SCORE_NAMES:
        NUM_FEATURE_NAMES.append(f"{_cam}_{_s}_variance")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_instability")
    NUM_FEATURE_NAMES.append(f"{_cam}_tear_peak_sharpness")

NUM_FEATURE_NAMES.extend(["cross_cam_tear_mean_diff", "cross_cam_tear_max_diff"])
NUM_FEATURE_NAMES.extend(["padding_fraction_cam1", "padding_fraction_cam2", "camera2_present"])
NUM_FEATURE_NAMES.extend(["speed", "grammage_weight", "web_width", "pap_len", "detector"])
CAT_COL_NAMES = ["printer", "grade", "paper_supplier"]

assert len(NUM_FEATURE_NAMES) == 68, f"Feature count mismatch: {len(NUM_FEATURE_NAMES)}"


# ── Frame helpers ─────────────────────────────────────────────────────────────

def is_valid_frame(frame: dict) -> bool:
    scores = frame.get("scores")
    if not isinstance(scores, dict):
        return False
    return any(v > 0 for v in scores.values())


def extract_score_vector(frame: dict) -> list:
    scores = frame.get("scores", {})
    return [float(scores.get(k, 0.0)) for k in SCORE_KEYS]


def get_valid_cameras(videos: list) -> list:
    valid_cams = []
    for v in videos:
        frames = v.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            continue
        usable = [fr for fr in frames if is_valid_frame(fr)]
        if len(usable) >= MIN_VALID_FRAMES:
            valid_cams.append(usable)
    return valid_cams


def extract_camera_window(frames: list) -> tuple[np.ndarray | None, int]:
    """
    Aligns a 300-frame window to the tear-score peak.
    Left-pads with zeros if the sequence is shorter than 300 frames.

    Returns:
        array (300, 5) and the valid (non-padded) length.
    """
    tear_scores = [fr.get("scores", {}).get("tear", 0.0) for fr in frames]
    if not tear_scores:
        return None, 0

    tear_idx    = int(np.argmax(tear_scores))
    seq_vectors = [extract_score_vector(fr) for fr in frames[:tear_idx + 1]]
    valid_length = len(seq_vectors)

    if valid_length > TIMESTEPS:
        seq_vectors  = seq_vectors[-TIMESTEPS:]
        valid_length = TIMESTEPS

    if valid_length < TIMESTEPS:
        pad         = [[0.0] * NUM_FEATURES_PER_CAM] * (TIMESTEPS - valid_length)
        seq_vectors = pad + seq_vectors

    return np.array(seq_vectors, dtype=np.float32), valid_length


def build_event_sequence(valid_cams: list) -> tuple[np.ndarray | None, dict | None]:
    """
    Concatenates up to 2 camera sequences into a (300, 10) array.
    Pads with zeros if only one camera is present.

    Returns:
        X_seq (300, 10) and a cam_meta dict, or (None, None) if no usable cameras.
    """
    cam_sequences, valid_lengths = [], []

    for i in range(min(len(valid_cams), 2)):
        seq, vlen = extract_camera_window(valid_cams[i])
        if seq is None:
            continue
        cam_sequences.append(seq)
        valid_lengths.append(vlen)

    if len(cam_sequences) == 2:
        camera2_present = 1
    elif len(cam_sequences) == 1:
        cam_sequences.append(np.zeros((TIMESTEPS, NUM_FEATURES_PER_CAM), dtype=np.float32))
        valid_lengths.append(0)
        camera2_present = 0
    else:
        return None, None

    X_seq = np.concatenate(cam_sequences, axis=1)  # (300, 10)
    cam_meta = {
        "valid_length_cam1":     valid_lengths[0],
        "valid_length_cam2":     valid_lengths[1],
        "padding_fraction_cam1": (TIMESTEPS - valid_lengths[0]) / TIMESTEPS,
        "padding_fraction_cam2": (TIMESTEPS - valid_lengths[1]) / TIMESTEPS,
        "camera2_present":       camera2_present,
    }
    return X_seq, cam_meta


# ── Metadata helpers ──────────────────────────────────────────────────────────

def parse_numeric(value) -> float:
    """Strips units from strings like '2.1 m/s' → 2.1."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        m = re.search(r"[-+]?\d*\.?\d+", str(value))
        if m:
            return float(m.group())
    except Exception:
        pass
    return np.nan


def parse_metadata(ev: dict) -> dict:
    return {
        "speed":           parse_numeric(ev.get("speed")),
        "grammage_weight": parse_numeric(ev.get("grammage_weight")),
        "web_width":       parse_numeric(ev.get("web_width")),
        "pap_len":         parse_numeric(ev.get("pap_len")),
        "detector":        parse_numeric(ev.get("detector")),
        "printer":         ev.get("printer"),
        "grade":           ev.get("grade"),
        "paper_supplier":  ev.get("paper_supplier"),
        "date_time_str":   ev.get("date_time_str"),
    }


# ── Feature engineering ───────────────────────────────────────────────────────

def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def _slope(x: np.ndarray) -> float:
    if np.std(x) < 1e-8:
        return 0.0
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])


def _instability(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(x))))


def extract_features(X_seq: np.ndarray, event_data: list) -> list:
    """
    Builds a flat feature vector per event.

    Output layout:
      58 camera features | 2 cross-cam | 3 cam-meta | 5 numeric-meta | 3 cat
      Total: 71 raw (68 numeric + 3 categorical strings)
    """
    features = []
    for i in range(len(X_seq)):
        seq  = X_seq[i]    # (300, 10)
        cam1 = seq[:, :5]
        cam2 = seq[:, 5:]
        feat = []

        for cam in [cam1, cam2]:
            feat.extend(np.mean(cam, axis=0).tolist())
            feat.extend(np.std(cam,  axis=0).tolist())
            feat.extend(np.max(cam,  axis=0).tolist())
            tear      = cam[:, 4]
            no_defect = cam[:, 0]
            feat.append(_slope(tear))
            feat.append(_slope(no_defect))
            feat.append(float(np.mean(tear[-50:])))
            feat.append(float(np.mean(tear[:50])))
            feat.append(float(np.mean(tear[-50:]) - np.mean(tear[:50])))
            ent = _entropy(cam)
            feat.append(float(np.mean(ent)))
            feat.append(float(np.std(ent)))
            feat.extend(np.var(cam, axis=0).tolist())
            feat.append(_instability(tear))
            feat.append(float(np.max(tear) - np.mean(tear[-50:])))

        # cross-camera
        cam1_tear = seq[:, 4]
        cam2_tear = seq[:, 9]
        feat.append(float(np.mean(cam1_tear) - np.mean(cam2_tear)))
        feat.append(float(np.max(cam1_tear)  - np.max(cam2_tear)))

        # camera meta
        cm = event_data[i]["cam_meta"]
        feat.append(cm["padding_fraction_cam1"])
        feat.append(cm["padding_fraction_cam2"])
        feat.append(cm["camera2_present"])

        # numeric metadata
        m = event_data[i]["metadata"]
        feat.append(m["speed"])
        feat.append(m["grammage_weight"])
        feat.append(m["web_width"])
        feat.append(m["pap_len"])
        feat.append(m["detector"])

        # categorical (raw strings — OHE applied in train/inference)
        feat.append(m["printer"])
        feat.append(m["grade"])
        feat.append(m["paper_supplier"])

        features.append(feat)
    return features


def split_num_cat(raw: list) -> tuple[np.ndarray, np.ndarray]:
    """Splits raw feature list into (X_num float64, X_cat object)."""
    X_num = np.array([r[:-3] for r in raw], dtype=float)
    X_cat = np.array([r[-3:] for r in raw])
    return X_num, X_cat


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _labels_from_zip(zip_path: str) -> dict:
    labels = {}
    with zipfile.ZipFile(zip_path) as z:
        lf = [n for n in z.namelist() if n.lower().endswith("labels.txt")]
        if not lf:
            return labels
        lines = z.open(lf[0]).read().decode("utf-8", errors="ignore").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[;,:\s]+", line)
        if len(parts) >= 2 and parts[-1] in ("0", "1"):
            labels[parts[0].replace("_event.json", "")] = int(parts[-1])
    return labels


def load_labelled_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Reads all *_events.zip files in data_dir.
    Each ZIP must contain a labels.txt file.

    Returns:
        X_seq  (N, 300, 10)
        y      (N,)
        event_data  list of dicts with event_id, label, cam_meta, metadata
    """
    zips = sorted(
        str(p) for p in Path(data_dir).glob("*_events.zip")
    )
    if not zips:
        raise FileNotFoundError(f"No *_events.zip files found in {data_dir}")

    X_list, y_list, event_data = [], [], []
    n_total = n_no_label = n_no_cam = 0

    for zip_path in zips:
        log.info("Loading %s", Path(zip_path).name)
        labels = _labels_from_zip(zip_path)

        with zipfile.ZipFile(zip_path) as z:
            jfiles = [n for n in z.namelist() if n.endswith("_event.json")]
            for jf in tqdm(jfiles, desc=Path(zip_path).name, leave=False):
                n_total += 1
                event_id = jf.split("/")[-1].replace("_event.json", "")
                try:
                    if event_id not in labels:
                        n_no_label += 1
                        continue
                    ev         = json.load(z.open(jf))
                    valid_cams = get_valid_cameras(ev.get("videos", []))
                    if not valid_cams:
                        n_no_cam += 1
                        continue
                    X_seq_ev, cam_meta = build_event_sequence(valid_cams)
                    if X_seq_ev is None:
                        n_no_cam += 1
                        continue
                    X_list.append(X_seq_ev)
                    y_list.append(labels[event_id])
                    event_data.append({
                        "event_id": event_id,
                        "label":    labels[event_id],
                        "cam_meta": cam_meta,
                        "metadata": parse_metadata(ev),
                    })
                except Exception as exc:
                    log.warning("Skipping %s — %s", event_id, exc)

    log.info(
        "Dataset: %d total | %d kept | %d no-label | %d no-camera",
        n_total, len(y_list), n_no_label, n_no_cam,
    )
    log.info("Class distribution: %s", np.bincount(np.array(y_list, dtype=int)).tolist())
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), event_data


def _parse_single_event(ev: dict, event_id: str) -> tuple:
    valid_cams = get_valid_cameras(ev.get("videos", []))
    if not valid_cams:
        raise ValueError("no valid cameras")
    X_seq_ev, cam_meta = build_event_sequence(valid_cams)
    if X_seq_ev is None:
        raise ValueError("could not build sequence")
    return X_seq_ev, cam_meta, parse_metadata(ev)


def load_new_events(input_dir: str) -> tuple[np.ndarray, list, list]:
    """
    Loads events for scoring (no labels required).
    Accepts *_events.zip files and loose *_event.json files.

    Returns:
        X_seq      (N, 300, 10)
        event_data list of dicts
        skipped    list of event_ids that failed to parse
    """
    zips  = sorted(glob(str(Path(input_dir) / "*_events.zip")))
    jsons = sorted(glob(str(Path(input_dir) / "*_event.json")))

    if not zips and not jsons:
        raise FileNotFoundError(
            f"No input files found in {input_dir}. "
            "Drop *_events.zip or *_event.json files there and re-run."
        )

    log.info("Found %d ZIP(s) and %d loose JSON(s)", len(zips), len(jsons))
    X_list, event_data, skipped = [], [], []

    for zip_path in zips:
        log.info("Reading %s", Path(zip_path).name)
        with zipfile.ZipFile(zip_path) as z:
            jfiles = [n for n in z.namelist() if n.endswith("_event.json")]
            for jf in tqdm(jfiles, desc=Path(zip_path).name, leave=False):
                event_id = jf.split("/")[-1].replace("_event.json", "")
                try:
                    ev = json.load(z.open(jf))
                    X_seq_ev, cam_meta, meta = _parse_single_event(ev, event_id)
                    X_list.append(X_seq_ev)
                    event_data.append({
                        "event_id": event_id,
                        "source":   Path(zip_path).name,
                        "cam_meta": cam_meta,
                        "metadata": meta,
                    })
                except Exception as exc:
                    log.warning("Skipping %s — %s", event_id, exc)
                    skipped.append(event_id)

    for json_path in tqdm(jsons, desc="Loose JSONs", leave=False):
        event_id = Path(json_path).name.replace("_event.json", "")
        try:
            with open(json_path, encoding="utf-8") as f:
                ev = json.load(f)
            X_seq_ev, cam_meta, meta = _parse_single_event(ev, event_id)
            X_list.append(X_seq_ev)
            event_data.append({
                "event_id": event_id,
                "source":   Path(json_path).name,
                "cam_meta": cam_meta,
                "metadata": meta,
            })
        except Exception as exc:
            log.warning("Skipping %s — %s", event_id, exc)
            skipped.append(event_id)

    if not X_list:
        raise RuntimeError("No events could be parsed.")

    log.info("Loaded %d events (%d skipped)", len(event_data), len(skipped))
    return np.array(X_list, dtype=np.float32), event_data, skipped
