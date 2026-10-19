import shutil

import numpy as np
import pytest
from PIL import Image

from roadseg.config import Config
from roadseg.data import ensure_disjoint, inspect_split, make_dataset, read_pair
from roadseg.demo import generate


@pytest.fixture
def config():
    return Config(height=32, width=64, batch_size=1)


@pytest.mark.parametrize(
    "change",
    [
        {"height": 33},
        {"classes": ["road", "road"]},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"backbone": "unknown"},
        {"imagenet_weights": True},
        {"epochs": True},
    ],
)
def test_config_rejects_invalid_values(change):
    with pytest.raises(ValueError):
        Config(**change)


def test_non_square_images_and_nearest_mask_resize(tmp_path, config):
    root = generate(tmp_path / "dataset")
    pairs = inspect_split(root, "train", config)
    image, mask, weights = read_pair(pairs[0].image, pairs[0].mask, config)
    assert image.shape == (32, 64, 3) and mask.shape == (32, 64)
    assert set(np.unique(mask)) == {0, 1} and weights.min() == 1
    batch = next(iter(make_dataset(pairs, config)))
    assert tuple(batch[0].shape) == (1, 32, 64, 3)


def test_masks_preserve_palette_indices_and_ignore_pixels(tmp_path, config):
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (64, 32), "green").save(image)
    values = np.zeros((32, 64), dtype=np.uint8)
    values[:, 20:40] = 1
    values[:8] = 255
    paletted = Image.fromarray(values).convert("P")
    paletted.putpalette([i for i in range(256) for _ in range(3)])
    paletted.save(mask)
    _, labels, weights = read_pair(image, mask, config)
    assert np.all(weights[:8] == 0) and np.all(labels[:8] == 0)
    assert np.all(labels[8:, 20:40] == 1)


def test_pairing_size_and_rgb_mask_validation(tmp_path, config):
    root = generate(tmp_path / "dataset")
    mask = root / "train/masks/shape-0.png"
    mask.unlink()
    with pytest.raises(ValueError, match="exactly one"):
        inspect_split(root, "train", config)
    Image.new("RGB", (96, 64), "red").save(mask)
    with pytest.raises(ValueError, match="single-channel"):
        inspect_split(root, "train", config)
    Image.new("L", (30, 30), 1).save(mask)
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_split(root, "train", config)


def test_unknown_labels_and_all_ignore_rejected(tmp_path, config):
    root = generate(tmp_path / "dataset")
    mask = root / "train/masks/shape-0.png"
    Image.new("L", (96, 64), 3).save(mask)
    with pytest.raises(ValueError, match="unknown class"):
        inspect_split(root, "train", config)
    Image.new("L", (96, 64), 255).save(mask)
    with pytest.raises(ValueError, match="only ignored"):
        inspect_split(root, "train", config)


def test_duplicate_frames_rejected_across_splits(tmp_path, config):
    root = generate(tmp_path / "dataset")
    shutil.copyfile(root / "train/images/shape-0.png", root / "val/images/shape-0.png")
    with pytest.raises(ValueError, match="multiple splits"):
        ensure_disjoint(
            inspect_split(root, "train", config), inspect_split(root, "val", config)
        )


def test_augmentation_keeps_images_labels_and_weights_aligned(tmp_path, config):
    # Red channel exactly encodes target class; flipping must preserve equality.
    root = tmp_path / "dataset"
    (root / "train/images").mkdir(parents=True)
    (root / "train/masks").mkdir()
    labels = np.zeros((32, 64), np.uint8)
    labels[:, :20] = 1
    rgb = np.zeros((32, 64, 3), np.uint8)
    rgb[:, :, 0] = labels * 255
    Image.fromarray(rgb).save(root / "train/images/frame.png")
    Image.fromarray(labels).save(root / "train/masks/frame.png")
    image, mask, _ = next(
        iter(make_dataset(inspect_split(root, "train", config), config, True))
    )
    np.testing.assert_array_equal(image.numpy()[0, :, :, 0] / 255, mask.numpy()[0])
