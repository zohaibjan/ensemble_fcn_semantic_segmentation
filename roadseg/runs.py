import hashlib
import json
from pathlib import Path

from .config import Config


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fresh_directory(path):
    path = Path(path).resolve()
    # Existing directories, including empty ones, require an explicit new name.
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train(data, output, config):
    import tensorflow as tf

    from .data import ensure_disjoint, inspect_split, make_dataset
    from .model import build_model, compile_model

    train_pairs = inspect_split(data, "train", config)
    val_pairs = inspect_split(data, "val", config)
    ensure_disjoint(train_pairs, val_pairs)
    tf.keras.utils.set_random_seed(config.seed)
    model = compile_model(build_model(config), config)
    directory = fresh_directory(output)
    write_json(directory / "status.json", {"status": "running"})
    write_json(directory / "config.json", config.to_dict())

    def details(pairs):
        return [
            {
                "image": str(p.image.relative_to(Path(data).resolve())),
                "mask": str(p.mask.relative_to(Path(data).resolve())),
                "image_pixels_sha256": p.image_sha256,
                "mask_file_sha256": p.mask_sha256,
            }
            for p in pairs
        ]

    write_json(
        directory / "dataset.json",
        {"train": details(train_pairs), "val": details(val_pairs)},
    )
    try:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(directory / "model.keras"), monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.CSVLogger(str(directory / "epochs.csv")),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            make_dataset(train_pairs, config, True),
            validation_data=make_dataset(val_pairs, config),
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
            shuffle=False,
        )
        clean = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }
        # NaN must not silently produce a supposedly usable checkpoint.
        write_json(directory / "history.json", clean)
        if not (directory / "model.keras").is_file():
            raise RuntimeError("Training did not produce a valid checkpoint")
        write_json(
            directory / "metadata.json",
            {
                "schema_version": 1,
                "config": config.to_dict(),
                "tensorflow": tf.__version__,
                "model_sha256": file_sha256(directory / "model.keras"),
                "selection": "best validation loss",
            },
        )
        write_json(
            directory / "status.json",
            {"status": "completed", "epochs": len(clean["loss"])},
        )
    except BaseException as exc:
        write_json(
            directory / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    return directory


def load_bundle(directory):
    import tensorflow as tf

    from . import model as registered_layers  # noqa: F401 - register VGGPreprocess for safe loading

    directory = Path(directory).resolve()
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("Only completed training runs can be loaded")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unknown model bundle schema")
    config = Config(**metadata["config"])
    model_path = directory / "model.keras"
    if file_sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("Model checksum does not match its metadata")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    if model.input_shape != (
        None,
        config.height,
        config.width,
        3,
    ) or model.output_shape != (None, config.height, config.width, len(config.classes)):
        raise ValueError("Saved model shape does not match its metadata")
    return model, config, metadata
