import argparse
import json
import os
from pathlib import Path

from .config import Config


def parser():
    cli = argparse.ArgumentParser(
        description="Train and evaluate modern roadside FCN segmentation"
    )
    commands = cli.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo-data", help="Create a small fictional CPU smoke-test dataset"
    )
    demo.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    pred = commands.add_parser("predict")
    pred.add_argument("--runs", nargs="+", required=True)
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--weights", nargs="+", type=float)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--runs", nargs="+", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--weights", nargs="+", type=float)
    fuse = commands.add_parser(
        "fuse-masks", help="Fuse explicitly mapped binary specialist PNG folders"
    )
    fuse.add_argument("--manifest", required=True)
    fuse.add_argument("--output", required=True)
    fuse.add_argument("--conflicts", choices=["error", "last"], default="error")
    fuse.add_argument(
        "--roi", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM")
    )
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    # Conservative CPU defaults; operators can supply their own values before launch.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        if args.command == "demo-data":
            from .demo import generate

            result = generate(args.output)
        elif args.command == "train":
            from .runs import train

            result = train(args.data, args.output, Config.load(args.config))
        elif args.command == "predict":
            from .inference import predict_folder

            result = predict_folder(args.runs, args.input, args.output, args.weights)
        elif args.command == "evaluate":
            from .inference import evaluate

            output = Path(args.output)
            if output.exists():
                raise ValueError(
                    "Evaluation output already exists; choose a new filename"
                )
            result = evaluate(args.runs, args.data, args.split, args.weights)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
        else:
            from .fusion import fuse_folders

            result = fuse_folders(args.manifest, args.output, args.conflicts, args.roi)
        print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"roadseg: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
