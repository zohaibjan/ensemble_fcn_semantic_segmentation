# Migration from the earlier research scripts

The source at commit `cd95fa091947c1bf5e54e10b7aac2ceb0e1faf8d` is preserved byte-for-byte under `legacy/`; `legacy/ORIGIN.json` records original Git blob hashes. Git history is retained.

| Earlier entry point | Modern replacement |
| --- | --- |
| `segmenter.py --model_path ...` restoring `model.ckpt` | `roadseg predict --runs <modern-run-dir> --input <images> --output <new-dir>` |
| `main.py` with hard-coded folders and import-time execution | `roadseg train`, `predict`, `evaluate` or `fuse-masks` |
| `fuse_models.fuse_seg` with filename-derived categories | `roadseg fuse-masks --manifest <explicit-class-map.json>` |
| `test_image_reader.py` / `scipy.misc` | `roadseg.images` and streaming `roadseg.data` |
| Implicit folder overwrite/delete | New output paths required |
| TF1 checkpoint plus MatConvNet VGG `.mat` | Self-contained `.keras` file plus config/provenance bundle |

The VGG19 model is an FCN8-style variant with current Keras preprocessing and a smaller configurable decoder. Its layer names, decoder tensors and preprocessing differ from the historical network. No automatic checkpoint conversion or numerical equivalence is claimed. The simplest supported migration is retraining on your authorised image/mask pairs.

For binary labels, use IDs 0 and 1. If a historical dataset stores foreground as 255, explicitly remap that value to 1 before training: this app reserves 255 for ignored/void pixels. RGB colour masks also need an explicit colour-to-ID mapping. The app deliberately avoids guessing either mapping.

The `distance*.py`, `lanePredict.py` and NVGX/Excel routines remain historical code under `legacy/`. Their camera calibration, proprietary input assumptions, missing helpers and physical-distance calculations have **not** been ported or validated. A calibration dataset and expected outputs are needed before reviving that part of the project. Running archived scripts may still use old APIs or delete paths; the modern CLI never imports them.

Create new train/val/test partitions by source drive or video. Keep the original historical results separately and clearly label any newly measured model runs with their data split and seed.
