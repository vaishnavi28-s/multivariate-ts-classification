"""Tests for preprocessing.py — frame validation, sequence building, feature extraction."""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    is_valid_frame, extract_score_vector, get_valid_cameras,
    extract_camera_window, build_event_sequence,
    parse_numeric, extract_features, split_num_cat,
    NUM_FEATURE_NAMES, TIMESTEPS, NUM_FEATURES_PER_CAM,
)


def _make_frame(scores: dict) -> dict:
    return {"name": "0001.jpg", "label": "no_defect", "scores": scores}


def _make_valid_frame(tear=0.01):
    return _make_frame({
        "no_defect": 0.90, "defect": 0.03,
        "rollenwechsel": 0.02, "Kantenfehler": 0.03, "tear": tear,
    })


# ── is_valid_frame ────────────────────────────────────────────────────────────

def test_valid_frame_passes():
    assert is_valid_frame(_make_valid_frame()) is True


def test_invalid_frame_all_zeros():
    assert is_valid_frame(_make_frame({"no_defect": 0.0, "defect": 0.0,
                                        "rollenwechsel": 0.0, "Kantenfehler": 0.0,
                                        "tear": 0.0})) is False


def test_invalid_frame_missing_scores():
    assert is_valid_frame({"name": "0001.jpg"}) is False


# ── extract_score_vector ──────────────────────────────────────────────────────

def test_score_vector_length():
    vec = extract_score_vector(_make_valid_frame())
    assert len(vec) == NUM_FEATURES_PER_CAM


def test_score_vector_order():
    frame = _make_frame({
        "no_defect": 0.9, "defect": 0.05,
        "rollenwechsel": 0.02, "Kantenfehler": 0.02, "tear": 0.01,
    })
    vec = extract_score_vector(frame)
    assert vec[0] == pytest.approx(0.9)
    assert vec[4] == pytest.approx(0.01)


def test_score_vector_missing_key_defaults_to_zero():
    frame = _make_frame({"no_defect": 0.9})
    vec = extract_score_vector(frame)
    assert vec[1] == pytest.approx(0.0)  # defect missing → 0


# ── extract_camera_window ─────────────────────────────────────────────────────

def test_camera_window_shape():
    frames = [_make_valid_frame(tear=float(i) / 100) for i in range(50)]
    arr, vlen = extract_camera_window(frames)
    assert arr is not None
    assert arr.shape == (TIMESTEPS, NUM_FEATURES_PER_CAM)
    assert vlen == 50


def test_camera_window_left_padding():
    # tear peak must be at the last frame so the full sequence is included
    frames = [_make_valid_frame(tear=0.01) for _ in range(19)]
    frames.append(_make_valid_frame(tear=0.99))   # peak at index 19
    arr, vlen = extract_camera_window(frames)
    assert arr.shape == (TIMESTEPS, NUM_FEATURES_PER_CAM)
    assert vlen == 20
    # first (300-20)=280 rows should be zero-padded
    assert np.all(arr[:280, :] == 0.0)


def test_camera_window_truncates_long_sequence():
    frames = [_make_valid_frame(tear=float(i) / 1000) for i in range(500)]
    arr, vlen = extract_camera_window(frames)
    assert arr.shape == (TIMESTEPS, NUM_FEATURES_PER_CAM)
    assert vlen == TIMESTEPS


# ── build_event_sequence ──────────────────────────────────────────────────────

def _make_camera(n_frames=50, tear_peak=0.8):
    frames = [_make_valid_frame() for _ in range(n_frames - 1)]
    frames.append(_make_valid_frame(tear=tear_peak))
    return frames


def test_two_cameras_shape():
    cams = [_make_camera(), _make_camera()]
    X, meta = build_event_sequence(cams)
    assert X is not None
    assert X.shape == (TIMESTEPS, 10)
    assert meta["camera2_present"] == 1


def test_one_camera_pads_second():
    cams = [_make_camera()]
    X, meta = build_event_sequence(cams)
    assert X is not None
    assert X.shape == (TIMESTEPS, 10)
    assert meta["camera2_present"] == 0
    assert np.all(X[:, 5:] == 0.0)  # second camera columns zero


def test_no_cameras_returns_none():
    X, meta = build_event_sequence([])
    assert X is None
    assert meta is None


# ── parse_numeric ─────────────────────────────────────────────────────────────

def test_parse_numeric_string_with_units():
    assert parse_numeric("2.1 m/s") == pytest.approx(2.1)
    assert parse_numeric("52 g/m2") == pytest.approx(52.0)
    assert parse_numeric("1639 m")  == pytest.approx(1639.0)


def test_parse_numeric_plain_float():
    assert parse_numeric(3.14) == pytest.approx(3.14)


def test_parse_numeric_none_returns_nan():
    assert np.isnan(parse_numeric(None))


# ── Feature count invariant ───────────────────────────────────────────────────

def test_feature_name_count():
    assert len(NUM_FEATURE_NAMES) == 68


def test_extract_features_shape():
    cams  = [_make_camera()]
    X, meta = build_event_sequence(cams)
    event_data = [{
        "event_id": "TEST",
        "cam_meta": meta,
        "metadata": {
            "speed": 2.1, "grammage_weight": 52.0,
            "web_width": 1630.0, "pap_len": 1639.0, "detector": 16,
            "printer": "M1", "grade": "A", "paper_supplier": "SUP1",
            "date_time_str": "20210705_083658",
        },
    }]
    raw = extract_features(X[np.newaxis, ...], event_data)
    X_num, X_cat = split_num_cat(raw)
    assert X_num.shape == (1, 68)
    assert X_cat.shape == (1, 3)
