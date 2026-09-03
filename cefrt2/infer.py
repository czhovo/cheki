import argparse
import json
from pathlib import Path

import torch

from cefrt2.runtime import ChekiEdgeFitRT


def main():
    parser = argparse.ArgumentParser(description="Frozen ChekiEdgeFit-RT v2 FP32 inference")
    parser.add_argument("input", type=Path, help="An image file or directory of images")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output file")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("Input does not exist")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in extensions and p.is_file()) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("No images found")
    torch.set_num_threads(args.threads)
    model = ChekiEdgeFitRT(args.weights_dir, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for index, path in enumerate(paths, 1):
            result = {"image": path.name, **model.predict(path)}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"{index}/{len(paths)} {path.name}: {len(result['predictions'])} predictions", flush=True)


if __name__ == "__main__":
    main()
