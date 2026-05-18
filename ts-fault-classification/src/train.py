"""
train.py
--------
Training pipeline: stratified CV, feature extraction, XGBoost,
threshold tuning, artefact saving.

Each fold saves one artefact:
    processed_data/xgb_models/xgb_fold_{n}.pkl
    → { model, scaler, ohe, threshold, auc, accuracy, fpr }

Fold 1 is the best-performing fold and is used for production inference.
"""

import os
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, log_loss, confusion_matrix,
    classification_report, roc_curve,
)
import xgboost as xgb

from .preprocessing import load_labelled_dataset, extract_features, split_num_cat
from .threshold import find_best_threshold
from .excel_export import write_coloured_excel

log = logging.getLogger(__name__)

SEED         = 42
N_SPLITS     = 3
VAL_SIZE     = 0.2
MODEL_DIR    = Path("./processed_data/xgb_models")

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=SEED,
    early_stopping_rounds=30,
    verbosity=0,
)


def train(data_dir: str, output_dir: str = "./processed_data") -> None:
    MODEL_DIR_RUN = Path(output_dir) / "xgb_models"
    MODEL_DIR_RUN.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("TRAINING — loading labelled dataset from %s", data_dir)
    log.info("=" * 60)

    X_seq, y, event_data = load_labelled_dataset(data_dir)

    log.info("Extracting features (%d events)...", len(y))
    raw_features     = extract_features(X_seq, event_data)
    X_all_num, X_all_cat = split_num_cat(raw_features)
    log.info("Numeric: %s  Categorical: %s", X_all_num.shape, X_all_cat.shape)

    skf     = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    results = []

    for fold_idx, (pool_idx, test_idx) in enumerate(skf.split(X_all_num, y), start=1):
        log.info("=" * 60)
        log.info("FOLD %d", fold_idx)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
        train_rel, val_rel = next(sss.split(X_all_num[pool_idx], y[pool_idx]))
        train_idx = pool_idx[train_rel]
        val_idx   = pool_idx[val_rel]

        X_tr_num, X_vl_num, X_te_num = (
            X_all_num[train_idx], X_all_num[val_idx], X_all_num[test_idx]
        )
        X_tr_cat, X_vl_cat, X_te_cat = (
            X_all_cat[train_idx], X_all_cat[val_idx], X_all_cat[test_idx]
        )
        y_tr, y_vl, y_te = y[train_idx], y[val_idx], y[test_idx]

        log.info("Train %s | Val %s | Test %s", X_tr_num.shape, X_vl_num.shape, X_te_num.shape)

        # one-hot encode — fit on train only
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_tr_ohe = ohe.fit_transform(X_tr_cat)
        X_vl_ohe = ohe.transform(X_vl_cat)
        X_te_ohe = ohe.transform(X_te_cat)

        # scale numerics — fit on train only
        scaler = StandardScaler()
        X_tr   = np.hstack([scaler.fit_transform(X_tr_num), X_tr_ohe])
        X_vl   = np.hstack([scaler.transform(X_vl_num),    X_vl_ohe])
        X_te   = np.hstack([scaler.transform(X_te_num),    X_te_ohe])

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        log.info("scale_pos_weight: %.2f", spw)

        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        p_vl = model.predict_proba(X_vl)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]

        auc       = roc_auc_score(y_te, p_te)
        test_loss = log_loss(y_te, p_te)
        thr       = find_best_threshold(y_vl, p_vl)
        y_pred    = (p_te >= thr).astype(int)
        cm        = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn + 1e-12)

        log.info("AUC: %.4f | Loss: %.4f | Threshold: %.3f | Acc: %.4f | FPR: %.4f",
                 auc, test_loss, thr, acc, fpr)
        log.info("\n%s", classification_report(
            y_te, y_pred,
            target_names=["machine_problem", "paper_problem"],
            digits=3, zero_division=0,
        ))

        # ROC curve saved to file
        fpr_vals, tpr_vals, _ = roc_curve(y_te, p_te)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_vals, tpr_vals, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC Curve — Fold {fold_idx}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(MODEL_DIR_RUN / f"xgb_fold_{fold_idx}_roc.png", dpi=120)
        plt.close(fig)

        # per-event predictions CSV + Excel
        fold_df = _build_predictions_df(
            event_ids=[event_data[i]["event_id"] for i in test_idx],
            fold_idx=fold_idx,
            y_true=y_te,
            p_prob=p_te,
            y_pred=y_pred,
            thr=thr,
        )
        fold_df.to_csv(MODEL_DIR_RUN / f"xgb_fold_{fold_idx}_predictions.csv", index=False)
        write_coloured_excel(fold_df, str(MODEL_DIR_RUN / f"xgb_fold_{fold_idx}_predictions.xlsx"))

        # save artefact
        artefact = {
            "fold":      fold_idx,
            "model":     model,
            "scaler":    scaler,
            "ohe":       ohe,
            "threshold": thr,
            "auc":       auc,
            "test_loss": test_loss,
            "accuracy":  acc,
            "fpr":       fpr,
        }
        pkl_path = MODEL_DIR_RUN / f"xgb_fold_{fold_idx}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artefact, f)
        log.info("Artefact saved → %s", pkl_path)

        results.append({"fold": fold_idx, "auc": auc, "loss": test_loss, "df": fold_df})

    # merged CSV
    merged = pd.concat([r["df"] for r in results], ignore_index=True)
    merged.to_csv(MODEL_DIR_RUN / "xgb_all_folds_predictions.csv", index=False)

    aucs = [r["auc"] for r in results]
    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("Fold AUCs : %s", [round(a, 4) for a in aucs])
    log.info("Mean AUC  : %.4f ± %.4f", np.mean(aucs), np.std(aucs))
    best = max(results, key=lambda r: r["auc"])
    log.info("Best fold : %d (AUC %.4f) — use this for production", best["fold"], best["auc"])

    _log_zone_evaluation(merged)


def _build_predictions_df(
    event_ids, fold_idx, y_true, p_prob, y_pred, thr
) -> pd.DataFrame:
    def _einschaetzung(p):
        if p < 0.30:  return "Keine Reklamation"
        if p < 0.70:  return "Unsicher"
        return "Reklamation"

    def _farbe(p):
        if p < 0.30:  return "grün"
        if p < 0.70:  return "gelb"
        return "rot"

    return pd.DataFrame({
        "event_id":        event_ids,
        "fold":            fold_idx,
        "true_label":      y_true,
        "true_label_str":  ["paper_problem" if v else "machine_problem" for v in y_true],
        "probability":     p_prob.round(6),
        "score_%":         [round(float(p) * 100, 2) for p in p_prob],
        "confidence_%":    [
            round(((float(p) - thr) / (1 - thr) * 100) if float(p) >= thr
                  else ((thr - float(p)) / thr * 100), 2)
            for p in p_prob
        ],
        "confident_class": ["paper_problem" if float(p) >= thr else "machine_problem" for p in p_prob],
        "einschaetzung":   [_einschaetzung(float(p)) for p in p_prob],
        "farbe":           [_farbe(float(p)) for p in p_prob],
        "prediction":      y_pred,
        "prediction_str":  ["paper_problem" if v else "machine_problem" for v in y_pred],
        "correct":         (y_pred == y_true).astype(int),
        "threshold":       thr,
    })


def _log_zone_evaluation(merged: pd.DataFrame) -> None:
    log.info("-" * 60)
    log.info("ZONE EVALUATION (%d held-out test events)", len(merged))
    for zone_name, farbe in [
        ("Keine Reklamation (gruen)", "grün"),
        ("Unsicher          (gelb) ", "gelb"),
        ("Reklamation       (rot)  ", "rot"),
    ]:
        sub = merged[merged["farbe"] == farbe]
        n = len(sub)
        if n == 0:
            continue
        n_paper   = (sub["true_label"] == 1).sum()
        n_machine = (sub["true_label"] == 0).sum()
        acc_zone  = sub["correct"].mean() * 100
        log.info(
            "%s | N=%d (%.1f%%)  true_paper=%.1f%%  true_machine=%.1f%%  accuracy=%.1f%%",
            zone_name, n, n / len(merged) * 100,
            n_paper / n * 100, n_machine / n * 100, acc_zone,
        )
    confident = merged[merged["farbe"] != "gelb"]
    if len(confident) > 0:
        log.info(
            "Green+Red only — accuracy %.1f%% on %.1f%% of events",
            confident["correct"].mean() * 100,
            len(confident) / len(merged) * 100,
        )
