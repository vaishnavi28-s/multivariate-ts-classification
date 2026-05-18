"""
cli.py
------
Command-line interface for training, scoring, and single-event prediction.

Usage:
  python -m src.cli train   --data_dir /path/to/labelled/zips
  python -m src.cli score   [--fold 1]
  python -m src.cli predict --event_json /path/to/event.json [--fold 1]
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OUTPUT_DIR = "./processed_data"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "ts-fault-classification — industrial fault classification pipeline\n\n"
            "  train   — train on labelled ZIP files, save fold artefacts\n"
            "  score   — score new monthly ZIPs / loose JSONs, write CSV + Excel\n"
            "  predict — quick single-event prediction\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="Train on labelled event ZIPs")
    t.add_argument("--data_dir",   required=True,
                   help="Directory containing labelled *_events.zip files")
    t.add_argument("--output_dir", default=OUTPUT_DIR,
                   help=f"Output directory  (default: {OUTPUT_DIR})")

    s = sub.add_parser("score", help="Score new events from input directory")
    s.add_argument("--input_dir",  default=f"{OUTPUT_DIR}/new_events/input",
                   help="Directory with new *_events.zip or *_event.json files")
    s.add_argument("--output_dir", default=f"{OUTPUT_DIR}/new_events/output",
                   help="Directory to write predictions")
    s.add_argument("--model_dir",  default=f"{OUTPUT_DIR}/xgb_models",
                   help="Directory containing xgb_fold_N.pkl artefacts")
    s.add_argument("--fold",       type=int, default=1,
                   help="Fold model to use  (default: 1 — best fold)")

    pr = sub.add_parser("predict", help="Score a single event JSON")
    pr.add_argument("--event_json", required=True,
                    help="Path to *_event.json file")
    pr.add_argument("--model_dir",  default=f"{OUTPUT_DIR}/xgb_models",
                    help="Directory containing xgb_fold_N.pkl artefacts")
    pr.add_argument("--fold",       type=int, default=1,
                    help="Fold model to use  (default: 1 — best fold)")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "train":
        from .train import train
        train(data_dir=args.data_dir, output_dir=args.output_dir)

    elif args.mode == "score":
        from .inference import score
        score(
            model_dir=args.model_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fold=args.fold,
        )

    elif args.mode == "predict":
        import pandas as pd
        from .inference import load_artefact, predict_single

        art    = load_artefact(args.model_dir, fold=args.fold)
        result = predict_single(args.event_json, art)

        print("\n── Prediction ──────────────────────────────────────")
        print(f"  Event        : {result['event_id']}")
        print(f"  Prediction   : {result['prediction']}  ({result['label_str']})")
        print(f"  Probability  : {result['probability']:.4f}")
        print(f"  Score        : {result['score_%']:.1f}%")
        print(f"  Zone         : {result['einschaetzung']}  ({result['farbe']})")
        print(f"  Threshold    : {result['threshold']:.3f}")
        print("────────────────────────────────────────────────────\n")

        out_path = Path(args.event_json).parent / f"{result['event_id']}_prediction.csv"
        pd.DataFrame([result]).to_csv(out_path, index=False)
        print(f"  Saved → {out_path}\n")


if __name__ == "__main__":
    main()
