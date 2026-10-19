from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_PIXELS = 16_000_000


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images in {folder}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Image stems must be unique, including across extensions")
    return files


def read_rgb(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds 16 megapixels")
        if image.getexif().get(274, 1) != 1:
            raise ValueError(
                "Bake EXIF rotation into image pixels and realign its mask before use"
            )
        return image.convert("RGB").copy()


def read_mask(path, num_classes, ignore_id=255):
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Mask exceeds 16 megapixels")
        values = np.asarray(image)
        if values.ndim != 2 or values.dtype.kind not in "ui":
            raise ValueError(
                "Masks must be single-channel or palette PNG class IDs, not RGB colours"
            )
        if not np.all((values < num_classes) | (values == ignore_id)):
            raise ValueError(f"Mask has unknown class IDs: {path}")
        if not np.any(values != ignore_id):
            raise ValueError(
                "A training/evaluation mask cannot contain only ignored pixels"
            )
        return values.astype(np.uint8)


def palette(num_classes):
    colours = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        colours[i] = ((37 * i + 40) % 256, (97 * i + 80) % 256, (157 * i + 120) % 256)
    return colours


def save_mask(mask, path):
    values = np.asarray(mask)
    if (
        values.ndim != 2
        or not np.issubdtype(values.dtype, np.integer)
        or values.min() < 0
        or values.max() > 254
    ):
        raise ValueError("Output mask must be an integer class-ID array in 0–254")
    Image.fromarray(values.astype(np.uint8)).save(path)


def overlay(image, mask, colours, alpha=0.45):
    values = np.asarray(image, dtype=np.float32)
    result = values.copy()
    fg = mask != 0
    result[fg] = (1 - alpha) * values[fg] + alpha * colours[mask[fg]]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
