import json
from pathlib import Path

import numpy as np
from PIL import Image

from .images import image_files, overlay, palette, read_mask, read_rgb, save_mask
from .metrics import Confusion
from .runs import fresh_directory, load_bundle, write_json


def normalized_weights(weights, count):
    values = (
        np.ones(count, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        values.shape != (count,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or values.sum() <= 0
    ):
        raise ValueError(
            "Provide one finite, nonnegative ensemble weight per model, with a positive sum"
        )
    return values / values.sum()


class Predictor:
    def __init__(self, runs, weights=None):
        if not runs:
            raise ValueError("At least one model bundle is required")
        loaded = [load_bundle(path) for path in runs]
        self.models = [value[0] for value in loaded]
        self.config = loaded[0][1]
        for _, config, _ in loaded[1:]:
            if (config.classes, config.height, config.width) != (
                self.config.classes,
                self.config.height,
                self.config.width,
            ):
                raise ValueError(
                    "Ensemble models must have identical class order and input dimensions"
                )
        self.weights = normalized_weights(weights, len(runs))
        self.provenance = [
            {"model_sha256": value[2]["model_sha256"], "weight": float(weight)}
            for value, weight in zip(loaded, self.weights)
        ]

    def predict(self, image):
        import tensorflow as tf

        config = self.config
        pixels = np.asarray(
            image.resize((config.width, config.height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )[None]
        probability = np.zeros(
            (config.height, config.width, len(config.classes)), dtype=np.float32
        )
        for model, weight in zip(self.models, self.weights):
            if not weight:
                continue
            logits = model(pixels, training=False)
            scores = np.asarray(tf.nn.softmax(logits, axis=-1)[0])
            if not np.isfinite(scores).all():
                raise ValueError("Model produced non-finite probabilities")
            probability += scores * weight
        mask = probability.argmax(-1).astype(np.uint8)
        # Class IDs, never probabilities or RGB, are resized back with nearest neighbour.
        return np.asarray(
            Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
        )


def predict_folder(runs, input_dir, output, weights=None):
    predictor = Predictor(runs, weights)
    files = image_files(input_dir)
    directory = fresh_directory(output)
    (directory / "masks").mkdir()
    (directory / "overlays").mkdir()
    write_json(directory / "status.json", {"status": "running"})
    try:
        for path in files:
            image = read_rgb(path)
            mask = predictor.predict(image)
            save_mask(mask, directory / "masks" / (path.stem + ".png"))
            overlay(image, mask, palette(len(predictor.config.classes))).save(
                directory / "overlays" / (path.stem + ".png")
            )
        write_json(
            directory / "manifest.json",
            {
                "classes": predictor.config.classes,
                "models": predictor.provenance,
                "source_files": [p.name for p in files],
                "resize": "bilinear RGB; nearest-neighbour class IDs",
                "fusion": "weighted probability mean; argmax ties use lowest class ID",
            },
        )
        write_json(
            directory / "status.json", {"status": "completed", "images": len(files)}
        )
    except BaseException as exc:
        write_json(directory / "status.json", {"status": "failed", "error": str(exc)})
        raise
    return directory


def evaluate(runs, data, split="test", weights=None):
    from .data import ensure_disjoint, inspect_split

    predictor = Predictor(runs, weights)
    config = predictor.config
    pairs = inspect_split(data, split, config)
    ensure_disjoint(pairs)
    # Reject exact train/validation frame reuse when evaluating an independent test split.
    # Evaluating val is allowed, but recorded explicitly as model-selection data.
    if split == "test":
        test_hashes = {p.image_sha256 for p in pairs}
        for run in runs:
            provenance = json.loads((Path(run) / "dataset.json").read_text())
            used = {
                p["image_pixels_sha256"]
                for key in ("train", "val")
                for p in provenance[key]
            }
            if used & test_hashes:
                raise ValueError("Test split reuses a training/validation frame")
    confusion = Confusion(config.classes, config.ignore_id)
    for pair in pairs:
        image = read_rgb(pair.image)
        prediction = predictor.predict(image)
        truth = read_mask(pair.mask, len(config.classes), config.ignore_id)
        confusion.update(truth, prediction)
    return confusion.result() | {
        "split": split,
        "images": len(pairs),
        "models": predictor.provenance,
        "model_selection_data": split == "val",
    }
