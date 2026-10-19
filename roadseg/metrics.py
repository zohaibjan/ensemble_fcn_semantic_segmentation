import numpy as np


class Confusion:
    def __init__(self, classes, ignore_id=255):
        self.classes = tuple(classes)
        self.ignore_id = ignore_id
        self.matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)

    def update(self, truth, prediction):
        truth = np.asarray(truth)
        prediction = np.asarray(prediction)
        if truth.shape != prediction.shape or truth.ndim != 2:
            raise ValueError("Evaluation masks must have identical 2-D shapes")
        if truth.dtype.kind not in "ui" or prediction.dtype.kind not in "ui":
            raise ValueError("Evaluation requires integer class IDs")
        n = len(self.classes)
        valid = truth != self.ignore_id
        if not np.all((truth[valid] >= 0) & (truth[valid] < n)) or not np.all(
            (prediction >= 0) & (prediction < n)
        ):
            raise ValueError("Unknown evaluation class ID")
        self.matrix += np.bincount(
            (truth[valid].astype(np.int64) * n + prediction[valid]).ravel(),
            minlength=n * n,
        ).reshape(n, n)

    def result(self):
        matrix = self.matrix
        tp = np.diag(matrix)
        union = matrix.sum(0) + matrix.sum(1) - tp
        pixels = int(matrix.sum())
        if not pixels:
            raise ValueError("No valid labelled pixels to evaluate")
        ious = np.divide(tp, union, out=np.zeros(len(tp), dtype=float), where=union > 0)
        present = union > 0
        return {
            "pixel_accuracy": float(tp.sum() / pixels),
            "mean_iou": float(ious[present].mean()),
            "foreground_mean_iou": float(ious[1:][present[1:]].mean())
            if present[1:].any()
            else None,
            "valid_pixels": pixels,
            "per_class": [
                {
                    "id": i,
                    "name": name,
                    "iou": float(ious[i]) if present[i] else None,
                    "support_pixels": int(matrix[i].sum()),
                }
                for i, name in enumerate(self.classes)
            ],
            "confusion_matrix": matrix.tolist(),
        }
