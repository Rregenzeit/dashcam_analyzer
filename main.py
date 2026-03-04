#!/usr/bin/env python3
# main.py - CLI entry point for the dashcam lane-change violation analyzer

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect lane-change violations (no blinker) in dashcam footage."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file (e.g. blackbox.mp4)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Directory for violation logs and clips (default: ./output)",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display real-time annotated preview window",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n/s/m/l/x  (default: n from config.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override model in config if specified
    if args.model is not None:
        import config as cfg
        cfg.YOLO_MODEL = f"yolov8{args.model}.pt"

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    from pipeline import Pipeline
    pipeline = Pipeline(
        input_path=args.input,
        output_dir=args.output,
        show=args.show,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
