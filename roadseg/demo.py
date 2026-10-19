"""Fictional coloured shapes for wiring checks, never a road-accuracy benchmark."""

import numpy as np
from PIL import Image

from .runs import fresh_directory


def generate(output):
    root = fresh_directory(output)
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        images = root / split / "images"
        masks = root / split / "masks"
        images.mkdir(parents=True)
        masks.mkdir()
        for i in range(count):
            pixels = rng.integers(10, 45, (64, 96, 3), dtype=np.uint8)
            labels = np.zeros((64, 96), dtype=np.uint8)
            left = int(rng.integers(15, 45))
            right = left + 25
            pixels[:, left:right] = [80, 150, 210]
            labels[:, left:right] = 1
            Image.fromarray(pixels).save(images / f"shape-{i}.png")
            Image.fromarray(labels).save(masks / f"shape-{i}.png")
    return root
