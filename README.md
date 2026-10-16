# Roadside Semantic Segmentation Experiments

A research prototype adapting fully convolutional networks to identify roadside attributes using AusRAP-oriented categories.

## Provenance

This work extends [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow). Preserve upstream attribution and review the original project's terms and setup instructions.

## Code map

- `main.py`, `segmenter.py`: segmentation workflow.
- `fuse_models.py`: model fusion experiments.
- `lanePredict.py`: lane prediction helper.
- `distance*.py`: distance-related experiments.
- `TensorflowUtils.py`: TensorFlow utilities.

## Status

Work-in-progress research code, as described in the original repository. Data, model checkpoints and a compatible legacy TensorFlow environment must be supplied or configured before use. This is not a packaged inference service, and the documentation cleanup has not reproduced its results.

[Profile and project directory](https://github.com/zohaibjan)
