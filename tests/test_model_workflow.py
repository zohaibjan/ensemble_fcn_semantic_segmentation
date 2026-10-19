import json
import shutil

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from roadseg.config import Config
from roadseg.demo import generate
from roadseg.inference import Predictor, evaluate, predict_folder
from roadseg.model import build_model, compile_model
from roadseg.runs import file_sha256, load_bundle, train


def test_real_gradient_update_and_serialization(tmp_path):
    tf.keras.utils.set_random_seed(42)
    config = Config(height=32, width=64, decoder_channels=8, learning_rate=0.01)
    model = compile_model(build_model(config), config)
    image = np.full((1, 32, 64, 3), 128, dtype=np.float32)
    labels = np.ones((1, 32, 64), dtype=np.int32)
    weights = np.ones_like(labels, dtype=np.float32)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial = float(loss(labels, model(image, training=False)))
    before = [v.numpy().copy() for v in model.trainable_weights]
    for _ in range(5):
        model.train_on_batch(image, labels, sample_weight=weights)
    final = float(loss(labels, model(image, training=False)))
    assert np.isfinite(final) and final < initial
    assert any(
        not np.array_equal(a, b.numpy())
        for a, b in zip(before, model.trainable_weights)
    )
    model.save(tmp_path / "model.keras")
    restored = tf.keras.models.load_model(
        tmp_path / "model.keras", compile=False, safe_mode=True
    )
    np.testing.assert_allclose(restored(image).numpy(), model(image).numpy(), atol=1e-6)


def test_vgg19_forward_and_safe_reload_without_downloads(tmp_path):
    config = Config(
        height=32, width=64, backbone="vgg19", decoder_channels=4, train_backbone=False
    )
    model = build_model(config)
    image = np.full((1, 32, 64, 3), 128, np.float32)
    result = model(image, training=False).numpy()
    assert result.shape == (1, 32, 64, 2) and np.isfinite(result).all()
    assert not model.get_layer("block1_conv1").trainable
    model.save(tmp_path / "vgg.keras")
    restored = tf.keras.models.load_model(
        tmp_path / "vgg.keras", compile=False, safe_mode=True
    )
    np.testing.assert_allclose(restored(image).numpy(), result, atol=1e-5)


def test_ignored_pixels_do_not_change_weighted_loss():
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="mean_with_sample_weight"
    )
    labels = np.array([[[1, 0]]])
    weights = np.array([[[1.0, 0.0]]])
    a = np.array([[[[0.0, 2.0], [0.0, 50.0]]]])
    b = np.array([[[[0.0, 2.0], [50.0, 0.0]]]])
    assert float(loss(labels, a, sample_weight=weights)) == pytest.approx(
        float(loss(labels, b, sample_weight=weights))
    )


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    data = generate(root / "data")
    config = Config(height=32, width=64, decoder_channels=8, epochs=1, batch_size=2)
    run = train(data, root / "run", config)
    return root, data, run


def test_training_predict_evaluate_round_trip(trained_bundle):
    root, data, run = trained_bundle
    model, config, metadata = load_bundle(run)
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert (run / "epochs.csv").is_file() and metadata["model_sha256"] == file_sha256(
        run / "model.keras"
    )
    out = predict_folder([run], data / "test/images", root / "predictions")
    masks = list((out / "masks").glob("*.png"))
    assert len(masks) == 2
    mask = np.asarray(Image.open(masks[0]))
    assert mask.shape == (64, 96) and mask.dtype == np.uint8
    metrics = evaluate([run], data)
    assert 0 <= metrics["mean_iou"] <= 1 and metrics["valid_pixels"] == 2 * 64 * 96
    assert metrics["model_selection_data"] is False
    # Repeating the same model with any positive weights preserves probabilities.
    one = Predictor([run])
    two = Predictor([run, run], [1, 3])
    image = Image.open(data / "test/images/shape-0.png").convert("RGB")
    np.testing.assert_array_equal(one.predict(image), two.predict(image))
    with pytest.raises(FileExistsError):
        predict_folder([run], data / "test/images", out)


def test_test_frame_leakage_rejected(trained_bundle, tmp_path):
    root, data, run = trained_bundle
    duplicate = tmp_path / "data"
    shutil.copytree(data, duplicate)
    shutil.copyfile(
        duplicate / "train/images/shape-0.png", duplicate / "test/images/shape-0.png"
    )
    with pytest.raises(ValueError, match="reuses"):
        evaluate([run], duplicate)


def test_model_checksum_mismatch_rejected(trained_bundle, tmp_path):
    _, _, run = trained_bundle
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    with (copy / "model.keras").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(copy)


def test_incompatible_ensemble_class_order_rejected(trained_bundle, tmp_path):
    _, _, run = trained_bundle
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    metadata = json.loads((copy / "metadata.json").read_text())
    metadata["config"]["classes"] = ["road", "background"]
    (copy / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="class order"):
        Predictor([run, copy])


def test_ensemble_averages_probabilities_instead_of_logits():
    predictor = Predictor.__new__(Predictor)
    predictor.config = Config(height=32, width=64)

    def constant(values):
        return lambda pixels, training=False: tf.broadcast_to(
            tf.constant(values, dtype=tf.float32), (1, 32, 64, 2)
        )

    # Weighted mean probabilities choose class 1; weighted mean raw logits would choose 0.
    predictor.models = [constant([4.6, 0.0]), constant([0.0, 1.386])]
    predictor.weights = np.array([1 / 3, 2 / 3])
    image = Image.new("RGB", (17, 29))
    assert np.all(predictor.predict(image) == 1)
