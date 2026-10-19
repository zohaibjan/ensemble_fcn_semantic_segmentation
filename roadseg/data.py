import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .images import image_files, read_mask, read_rgb


@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path
    image_sha256: str
    mask_sha256: str


def inspect_split(root, split, config):
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    root = Path(root).resolve() / split
    images = image_files(root / "images")
    masks = root / "masks"
    expected = {p.stem + ".png" for p in images}
    if (
        not masks.is_dir()
        or {p.name for p in masks.iterdir() if p.is_file()} != expected
    ):
        raise ValueError(
            f"Every {split} image requires exactly one mask with the same stem and .png extension"
        )
    pairs = []
    for path in images:
        mask = masks / (path.stem + ".png")
        rgb = read_rgb(path)
        labels = read_mask(mask, len(config.classes), config.ignore_id)
        if rgb.size != (labels.shape[1], labels.shape[0]):
            raise ValueError(f"Image/mask size mismatch: {path.name}")
        # Hash decoded RGB pixels to catch the same frame saved with different metadata.
        identity = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        pairs.append(
            Pair(path, mask, identity, hashlib.sha256(mask.read_bytes()).hexdigest())
        )
    return pairs


def ensure_disjoint(*splits):
    seen = set()
    for split in splits:
        hashes = {p.image_sha256 for p in split}
        if hashes & seen:
            raise ValueError(
                "Identical image pixels appear in multiple splits; split by source video/site before training"
            )
        if len(hashes) != len(split):
            raise ValueError("A split contains duplicate image pixels")
        seen.update(hashes)


def read_pair(image_path, mask_path, config):
    image = np.asarray(
        read_rgb(image_path).resize(
            (config.width, config.height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    labels = read_mask(mask_path, len(config.classes), config.ignore_id)
    labels = np.asarray(
        Image.fromarray(labels).resize(
            (config.width, config.height), Image.Resampling.NEAREST
        ),
        dtype=np.int32,
    )
    weights = (labels != config.ignore_id).astype(np.float32)
    labels = np.where(weights > 0, labels, 0).astype(np.int32)
    return image, labels, weights


def make_dataset(pairs, config, training=False):
    import tensorflow as tf

    if not pairs:
        raise ValueError("Dataset is empty")
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(p.image) for p in pairs], [str(p.mask) for p in pairs])
    )
    if training:
        dataset = dataset.shuffle(
            len(pairs), seed=config.seed, reshuffle_each_iteration=True
        )

    def load(image_path, mask_path):
        def python_load(a, b):
            return read_pair(a.decode(), b.decode(), config)

        image, labels, weights = tf.numpy_function(
            python_load, [image_path, mask_path], [tf.float32, tf.int32, tf.float32]
        )
        image.set_shape((config.height, config.width, 3))
        labels.set_shape((config.height, config.width))
        weights.set_shape((config.height, config.width))
        return image, labels, weights

    dataset = dataset.map(load, num_parallel_calls=1)
    if training and config.augment:

        def augment(index, values):
            image, labels, weights = values
            seed = tf.stack([tf.cast(config.seed, tf.int32), tf.cast(index, tf.int32)])
            image = tf.image.stateless_random_flip_left_right(image, seed)
            labels = tf.squeeze(
                tf.image.stateless_random_flip_left_right(labels[..., None], seed), -1
            )
            weights = tf.squeeze(
                tf.image.stateless_random_flip_left_right(weights[..., None], seed), -1
            )
            return image, labels, weights

        dataset = dataset.enumerate().map(augment, num_parallel_calls=1)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    return dataset.with_options(options).batch(config.batch_size).prefetch(1)
