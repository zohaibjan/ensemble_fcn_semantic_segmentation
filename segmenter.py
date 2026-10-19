"""Predict using modern model bundles; old TF1 checkpoint flags are in legacy/."""

import sys

from roadseg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))
