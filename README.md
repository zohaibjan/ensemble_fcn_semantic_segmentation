# Roadside FCN Semantic Segmentation

A modern **TensorFlow 2.21 / Keras 3** implementation for roadside image segmentation, training, evaluation and ensemble prediction. The original research scripts are preserved under [`legacy/`](legacy/).

This project grew from fully convolutional network experiments for AusRAP-oriented roadside categories. The modern implementation provides a portable, tested workflow; it does not reproduce the historical model or its results.

[Zohaib's portfolio](https://zohaibjan.github.io/) · [GitHub profile](https://github.com/zohaibjan)

## What changed

| Earlier code | Modern workflow |
| --- | --- |
| TensorFlow 1 sessions, placeholders and `tf.flags` | TensorFlow 2 / Keras models and argparse CLI |
| Removed `scipy.misc` image functions | Pillow RGB loading and nearest-neighbour label resizing |
| Hard-coded Windows paths and missing helper imports | Explicit dataset, run and output paths |
| Whole-folder image loading | Streaming `tf.data` pipeline, path shuffling and bounded prefetch |
| Script deletes an existing output folder | Existing run/prediction directories are rejected |
| Filename-substring class guesses and folder-order overwrites | Explicit specialist class IDs and overlap policy |
| No active training/evaluation entry point | Training, checkpointing, prediction, per-class IoU and confusion matrices |
| No environment or automated tests | Installable package, synthetic tests and CPU CI |

**Old TensorFlow 1 checkpoints cannot be loaded directly.** The new FCN8-style models use a different decoder and preprocessing contract. Retrain with labelled masks or undertake a separately validated checkpoint-conversion project. The historical scripts and their attribution remain intact.

## Install

Use Python 3.11 or 3.12 in a fresh environment. The CPU workflow uses TensorFlow 2.21.0; no model weights are downloaded by default.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu,test]'
```

On Windows, activate with `.venv\Scripts\activate`. For a GPU environment, install `.[tensorflow,test]` instead of the CPU extra and follow [TensorFlow's official installation guide](https://www.tensorflow.org/install/pip) for your platform. GPU execution has not been validated in this modernisation. Do not install both CPU and GPU TensorFlow packages in one environment.

## Run a complete CPU example

The bundled generator creates fictional coloured shapes. This tests the software workflow, not road-scene accuracy.

```bash
roadseg demo-data --output data/demo
roadseg train --data data/demo --config examples/compact.json --output runs/demo
roadseg predict --runs runs/demo --input data/demo/test/images --output predictions/demo
roadseg evaluate --runs runs/demo --data data/demo --split test --output runs/demo/test-metrics.json
```

Run from the repository root. Every output directory above must be new; rerunning requires a new name. `python main.py ...` also accepts these subcommands. `python segmenter.py --runs runs/demo --input ... --output ...` is a prediction shortcut.

A completed training run contains:

- `model.keras`: checkpoint with the best validation loss, including preprocessing.
- `config.json` and `metadata.json`: ordered class names, input size, architecture and model checksum.
- `dataset.json`: relative source paths and image/mask hashes for split checks.
- `epochs.csv` and `history.json`: measured training and validation history.
- `status.json`: completed or failed state. Incomplete runs cannot be loaded for prediction.

Model loading uses Keras safe mode, but only load bundles you trust. A checksum detects changes relative to the accompanying metadata; it is not an authenticity signature.

## Use your own images and masks

```text
dataset/
  train/
    images/frame-a.jpg
    masks/frame-a.png
  val/
    images/frame-b.jpg
    masks/frame-b.png
  test/
    images/frame-c.jpg
    masks/frame-c.png
```

Each split requires matching image/mask stems. Images may be JPEG, PNG or BMP. Masks are single-channel or palette PNGs containing **integer class IDs**, not RGB colours. Set `classes` in order: class 0 is background; subsequent names map to IDs 1, 2, etc. Label **255 is ignored**, and its pixels are excluded from loss/metrics. Historical binary masks using 255 as foreground must be explicitly converted to class ID 1 before training; specialist mask fusion has a separate explicit encoding option.

The loader validates mask IDs, image/mask geometry, nonempty valid labels and duplicate decoded image pixels. It rejects incomplete pairs and additional files in the masks folder. Image dimensions must be at most 16 megapixels. Bake EXIF rotation into source pixels and align masks before use; rotated metadata is rejected rather than silently misaligning labels.

Training resizes RGB with bilinear interpolation and labels with nearest neighbour. Optional horizontal flips apply identically to RGB, labels and ignored-pixel weights. Height and width must each be divisible by 32. Images are streamed; preprocessing does not cache the full image corpus in RAM.

**Split by source video, drive or site before using this app.** Exact frame hashing catches duplicate pixels; it does not detect adjacent or near-duplicate frames from the same journey. A held-out test split must remain separate from both training and model-selection validation data. Dataset files should remain unchanged throughout a training run.

## Models and preprocessing

- **`compact`**: a small five-stage convolutional encoder for CPU development and plumbing checks.
- **`vgg19`**: a Keras VGG19 encoder with class-score skip connections at strides 8/16/32 and a configurable smaller decoder. This is FCN8-style, not the original MatConvNet/4096-channel network.
- Both return per-pixel logits at configured resolution. The saved model includes preprocessing: compact uses RGB/255; VGG19 uses its required RGB-to-BGR mean subtraction.
- `imagenet_weights: false` is the default. Set it to `true` for VGG19 only if you deliberately want Keras to download ImageNet weights. That download path has not been exercised here.
- `train_backbone: false` freezes the encoder. Training an uninitialised frozen encoder is generally only useful for wiring checks; use suitable pretrained weights for transfer-learning experiments.

Copy `examples/vgg19.json`, configure your class names and training parameters, then select that file with `--config`. Set your own epoch count and learning rate based on real validation data. The CPU defaults limit TensorFlow inter/intra-op threads to one unless you set `TF_NUM_INTEROP_THREADS` / `TF_NUM_INTRAOP_THREADS` before launch.

## Ensemble prediction

Average **probabilities**, not hard labels, for multiple modern models trained on the same ordered classes and input dimensions:

```bash
roadseg predict --runs runs/seed-42 runs/seed-99 --weights 1 1 \
  --input dataset/test/images --output predictions/ensemble
roadseg evaluate --runs runs/seed-42 runs/seed-99 --weights 1 1 \
  --data dataset --split test --output runs/ensemble-test.json
```

Models are loaded once, and images are processed individually. Memory grows with the number/size of loaded models. Weights must be finite and nonnegative with a positive sum. Choose weights on validation data before looking at test performance. Ensembles must have exactly matching class order and input size. Lowest class ID wins an exact argmax tie.

Outputs include raw class-ID PNGs at the original image dimensions, coloured overlays and a provenance manifest. Evaluation reports pixel accuracy, mean IoU, foreground mean IoU, class support and a confusion matrix. Classes absent from both truth and prediction have `iou: null` and are excluded from the mean. Background is included in `mean_iou`; `foreground_mean_iou` excludes ID 0.

## Fuse existing binary specialist masks

The legacy project combined separate line/pole/sign masks. You can still fuse those PNG outputs without restoring their old TensorFlow models. Write a manifest, with folder paths relative to the manifest:

```json
[
  {"folder": "line-masks", "class_id": 1, "foreground_value": 1},
  {"folder": "pole-masks", "class_id": 2, "foreground_value": 255}
]
```

```bash
roadseg fuse-masks --manifest specialists.json --output predictions/fused
# If overlap is intentional, later manifest entries take priority:
roadseg fuse-masks --manifest specialists.json --output predictions/fused-priority --conflicts last
# Optional normalised ROI: left, top, right, bottom
roadseg fuse-masks --manifest specialists.json --output predictions/fused-roi --roi 0 0.4 1 1
```

Folders must have identical PNG filename sets and image dimensions. This path performs vectorised mask operations, supports non-square images and rejects overlaps by default. It does not calibrate specialist probabilities or infer class IDs from filenames.

## Verification and limits

```bash
python -m pytest -q
```

The tests exercise **actual TensorFlow CPU computation**: gradient updates with loss reduction, compact and VGG19 inference, safe model save/reload, a small training→prediction→evaluation run, probability ensembles, mask validation, ignore pixels, duplicate-frame rejection, metrics, specialist fusion and protection of existing output directories. No API keys, original datasets or historical checkpoints are needed.

The compact example has also been run through the installed CLI. VGG19 was checked with random weights on CPU, including save/reload; full VGG19 training, GPU execution, ImageNet downloads and real roadside quality remain unvalidated. Synthetic shape scores are not evidence of road-detection performance.

See [migration notes](docs/MIGRATION.md), [legacy source provenance](legacy/ORIGIN.json) and [validation notes](docs/VALIDATION.md).

## Attribution

The original project extends [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow). Its original files and attribution are preserved under `legacy/` without rewriting their contents. The modern package is newly implemented, drawing on the FCN encoder/skip-decoder approach and current Keras APIs. No original data, pretrained weights or model-quality claims are bundled. Existing source/license provenance is retained; this change does not introduce a blanket relicensing of upstream code.

Primary implementation references: [TensorFlow segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation), [Keras VGG19](https://keras.io/api/applications/vgg/), [weighted loss reductions](https://keras.io/api/losses/) and [segmentation metrics](https://keras.io/api/metrics/segmentation_metrics/).
