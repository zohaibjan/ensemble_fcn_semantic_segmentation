"""Vectorized fusion of binary specialist masks with explicit class mapping."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .images import save_mask
from .runs import fresh_directory, write_json


def fuse_binary_arrays(masks, class_ids, conflicts="error", roi=None):
    if not masks or len(masks) != len(class_ids):
        raise ValueError("Each specialist mask needs a class ID")
    if conflicts not in {"error", "last"}:
        raise ValueError("conflicts must be error or last")
    if any(type(i) is not int or not 1 <= i <= 254 for i in class_ids) or len(
        set(class_ids)
    ) != len(class_ids):
        raise ValueError("Specialist class IDs must be unique integers in 1–254")
    shape = np.asarray(masks[0]).shape
    if len(shape) != 2:
        raise ValueError("Masks must be 2-D")
    output = np.zeros(shape, dtype=np.uint8)
    for mask, class_id in zip(masks, class_ids):
        mask = np.asarray(mask)
        if (
            mask.shape != shape
            or mask.dtype.kind not in "bui"
            or not np.isin(mask, [0, 1]).all()
        ):
            raise ValueError("Specialist masks must be matching 0/1 arrays")
        foreground = mask == 1
        if conflicts == "error" and np.any(foreground & (output != 0)):
            raise ValueError(
                "Specialist masks overlap; review them or choose explicit last-wins priority"
            )
        output[foreground] = class_id
    if roi is not None:
        if len(roi) != 4 or not all(np.isfinite(v) for v in roi):
            raise ValueError("ROI is left,top,right,bottom in 0–1")
        left, top, right, bottom = roi
        if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
            raise ValueError("Invalid ROI bounds")
        height, width = shape
        keep = np.zeros(shape, dtype=bool)
        keep[
            int(top * height) : int(bottom * height),
            int(left * width) : int(right * width),
        ] = True
        output[~keep] = 0
    return output


def fuse_folders(manifest, output, conflicts="error", roi=None):
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text())
    if not isinstance(spec, list) or not spec:
        raise ValueError("Manifest must be a nonempty list of specialist folders")
    folders = []
    ids = []
    foreground_values = []
    for entry in spec:
        if not isinstance(entry, dict) or not {"folder", "class_id"} <= set(entry):
            raise ValueError("Each specialist needs folder and class_id")
        if not isinstance(entry["folder"], str):
            raise ValueError("Specialist folder must be a path string")
        if set(entry) - {"folder", "class_id", "foreground_value"}:
            raise ValueError("Unknown specialist manifest key")
        folder = (manifest.parent / entry["folder"]).resolve()
        if not folder.is_dir():
            raise ValueError(f"Missing specialist folder: {folder}")
        value = entry.get("foreground_value", 1)
        if type(value) is not int or value not in {1, 255}:
            raise ValueError("foreground_value must be explicitly 1 or 255")
        folders.append(folder)
        ids.append(entry["class_id"])
        foreground_values.append(value)
    names = sorted(p.name for p in folders[0].glob("*.png"))
    if not names or any(
        set(p.name for p in folder.glob("*.png")) != set(names) for folder in folders
    ):
        raise ValueError(
            "Specialist folders must have identical nonempty PNG filename sets"
        )
    directory = fresh_directory(output)
    write_json(directory / "status.json", {"status": "running"})
    try:
        for name in names:
            masks = []
            for folder, value in zip(folders, foreground_values):
                with Image.open(folder / name) as image:
                    mask = np.asarray(image)
                if mask.ndim != 2 or not np.isin(mask, [0, value]).all():
                    raise ValueError(
                        "Mask is not binary under the declared foreground encoding"
                    )
                masks.append((mask == value).astype(np.uint8))
            save_mask(fuse_binary_arrays(masks, ids, conflicts, roi), directory / name)
        write_json(
            directory / "manifest.json",
            {"specialists": spec, "conflicts": conflicts, "roi": roi, "images": names},
        )
        write_json(directory / "status.json", {"status": "completed"})
    except BaseException as exc:
        write_json(directory / "status.json", {"status": "failed", "error": str(exc)})
        raise
    return directory
