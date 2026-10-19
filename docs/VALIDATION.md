# Validation record

The modern package was exercised locally on Linux, Python 3.12.14, `tensorflow-cpu 2.21.0`, Keras 3.15.1, NumPy 2.5.4 and Pillow 12.3.0. CPU-only GitHub Actions also checks Python 3.11 and 3.12. Check the latest workflow result for the exact published commit.

## Actual computation covered

- Compact model: real forward/backward updates with finite gradients and reduced loss, not a mocked training call.
- Compact model: safe `.keras` save/reload with matching logits.
- VGG19: random-weight forward inference at 32×64, encoder freezing, safe save/reload and matching outputs, with no weight download.
- Training: real one-epoch `fit`, validation, best-checkpoint selection, history/provenance files and subsequent model loading.
- Prediction/evaluation: original-size ID masks, overlays, exact confusion-matrix accounting, ignored pixels and independent test-frame checks.
- Ensemble: matching-model equivalence, incompatible class-order rejection, invalid weights and a case distinguishing probability averaging from logit averaging.
- Data: missing pairs, misaligned image sizes, RGB mask rejection, palette indices, invalid labels, all-ignored masks, duplicate image pixels and aligned augmentation.
- Specialist masks: explicit 0/1 or 0/255 encoding, overlap policy, non-square ROI, missing frames and non-destructive output handling.
- CLI: installed-command help, demo generation and a separate manual train→predict→evaluate sequence.

## Fictional shape example

The installed CLI was run with `examples/compact.json`: 6 train images, 2 validation images and 2 independent fictional shape images for a wiring check. Ten training epochs ran; the selected checkpoint came from the lowest validation loss. Training loss changed from 0.6473 to 0.0608 and validation loss reached 0.0371.

On those **two synthetic shape images only**, the selected checkpoint produced mean IoU 0.9215 and foreground IoU 0.8843. These numbers merely show that the pipeline can learn its artificial fixture. They are not roadside accuracy, a useful generalisation benchmark, or a comparison with the historical model. Do not use these numbers in deployment/business claims.

## Still requires real acceptance work

- Authorised real roadside data and subject-matter-reviewed class masks, split by drive/video/site.
- Full VGG19 training and hyperparameter selection on real validation data.
- Independent real-world test metrics, class imbalance/rare-class assessment and comparison of single models versus ensembles.
- GPU correctness, memory use, throughput and repeatability on the target hardware.
- Any ImageNet-weight download/transfer-learning path.
- TF1 checkpoint conversion and numerical equivalence, if historical weights become available.
- Physical camera calibration, NVGX parsing and distance/Excel workflows in the archived code.

Keras 3.15.1 emits NumPy `__array__(copy=...)` deprecation warnings during model serialization in this environment. They do not fail the tested round trips; no project warning filter is used to hide them. The tests do not contact AI services or download pretrained weights.
