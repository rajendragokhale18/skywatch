"""Evaluate a trained SkyWatch detector on the clean MBAOD test split."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--data",
        default="data/airborne_dataset/mbaod_yolo.yaml",
        help="Dataset YAML path",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device, e.g. mps or cpu",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size",
    )
    parser.add_argument(
        "--project",
        default="results/evaluation",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--name",
        default="clean_test",
        help="Name of this evaluation run",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )

    output_dir = Path(metrics.save_dir)
    box_metrics = metrics.box

    summary = {
        "model": str(model_path),
        "dataset": str(data_path),
        "split": "test",
        "image_size": args.imgsz,
        "device": args.device,
        "metrics": {
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
        },
        "per_class": {},
        "output_dir": str(output_dir.resolve()),
    }

    names = model.names
    for index, class_name in enumerate(names.values()):
        if index >= len(box_metrics.p) or index >= len(box_metrics.r):
            break
        summary["per_class"][str(class_name)] = {
            "precision": float(box_metrics.p[index]),
            "recall": float(box_metrics.r[index]),
            "map50": float(box_metrics.ap50[index]),
            "map50_95": float(box_metrics.ap[index].mean()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
