import json

import numpy as np
import pytest
from PIL import Image
from roadseg.fusion import fuse_binary_arrays, fuse_folders
from roadseg.inference import normalized_weights
from roadseg.metrics import Confusion


def test_confusion_metrics_with_ignored_and_absent_classes():
    metric = Confusion(["background", "road", "pole"])
    metric.update(
        np.array([[0, 1], [1, 255]], np.uint8), np.array([[0, 0], [1, 2]], np.uint8)
    )
    result = metric.result()
    assert result["valid_pixels"] == 3
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)
    assert result["mean_iou"] == pytest.approx(0.5)
    assert result["per_class"][2]["iou"] is None
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]


def test_invalid_evaluation_masks_fail():
    metric = Confusion(["background", "road"])
    with pytest.raises(ValueError):
        metric.update(np.array([[5]], np.uint8), np.array([[0]], np.uint8))
    with pytest.raises(ValueError):
        metric.result()


def test_non_square_fusion_roi_and_conflict_policy():
    a = np.zeros((4, 8), np.uint8)
    a[:, 1] = 1
    b = np.zeros_like(a)
    b[2:, 5] = 1
    fused = fuse_binary_arrays([a, b], [1, 2], roi=(0, 0.5, 1, 1))
    assert np.all(fused[:2] == 0) and fused[3, 1] == 1 and fused[3, 5] == 2
    with pytest.raises(ValueError, match="overlap"):
        fuse_binary_arrays([a, a], [1, 2])
    assert fuse_binary_arrays([a, a], [1, 2], conflicts="last")[0, 1] == 2


@pytest.mark.parametrize("weights", [[0, 0], [1, -1], [float("nan"), 1], [1]])
def test_ensemble_weights_reject_invalid(weights):
    with pytest.raises(ValueError):
        normalized_weights(weights, 2)


def test_normalized_weights():
    np.testing.assert_allclose(normalized_weights([1, 3], 2), [0.25, 0.75])


def test_explicit_folder_mapping_and_no_destructive_overwrite(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    x = np.zeros((4, 8), np.uint8)
    x[:, 1] = 255
    y = np.zeros_like(x)
    y[:, 5] = 1
    Image.fromarray(x).save(a / "frame.png")
    Image.fromarray(y).save(b / "frame.png")
    spec = tmp_path / "specialists.json"
    spec.write_text(
        json.dumps(
            [
                {"folder": "a", "class_id": 1, "foreground_value": 255},
                {"folder": "b", "class_id": 2},
            ]
        )
    )
    out = fuse_folders(spec, tmp_path / "out")
    result = np.asarray(Image.open(out / "frame.png"))
    assert result[0, 1] == 1 and result[0, 5] == 2
    with pytest.raises(FileExistsError):
        fuse_folders(spec, out)
    assert (out / "frame.png").is_file()


def test_missing_specialist_frames_rejected(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    Image.new("L", (8, 4), 1).save(a / "a.png")
    Image.new("L", (8, 4), 1).save(b / "b.png")
    spec = tmp_path / "specialists.json"
    spec.write_text(
        json.dumps([{"folder": "a", "class_id": 1}, {"folder": "b", "class_id": 2}])
    )
    with pytest.raises(ValueError, match="filename sets"):
        fuse_folders(spec, tmp_path / "out")
