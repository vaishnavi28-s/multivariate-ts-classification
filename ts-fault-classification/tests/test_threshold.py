"""Tests for threshold.py — decision boundary tuning."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.threshold import find_best_threshold


def test_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = find_best_threshold(y_true, y_prob)
    # any threshold between 0.3 and 0.7 is correct
    assert 0.3 <= thr <= 0.7


def test_threshold_in_valid_range():
    rng    = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    thr    = find_best_threshold(y_true, y_prob)
    assert 0.05 <= thr <= 0.95


def test_all_negative_class():
    y_true = np.zeros(100, dtype=int)
    y_prob = np.random.default_rng(0).uniform(0, 1, 100)
    # should not crash, returns a float
    thr = find_best_threshold(y_true, y_prob)
    assert isinstance(thr, float)


def test_highly_imbalanced():
    # mirrors the real dataset: ~8.5% positive
    rng    = np.random.default_rng(99)
    n      = 1000
    y_true = (rng.uniform(0, 1, n) < 0.085).astype(int)
    # positive class scores slightly higher
    y_prob = rng.uniform(0, 0.6, n)
    y_prob[y_true == 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    thr = find_best_threshold(y_true, y_prob)
    assert 0.05 <= thr <= 0.95
