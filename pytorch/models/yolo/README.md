# YOLOv1 Module Overview

This package contains a self-contained implementation of the YOLOv1 object detection model, including training, inference, dataset utilities, and auxiliary helpers. Use this document to understand what each file does, how to launch the training and inference workflows, and how to validate changes.

## Source Layout

| File | Purpose |
| --- | --- |
| `__init__.py` | Exposes the top-level package namespace for the YOLO implementation. |
| `config_yolov1.yaml` | Consolidated configuration covering model hyperparameters, training knobs, dataset paths, and inference thresholds. |
| `dataset.py` | `CocoDataset` wrapper that loads COCO-style annotations and produces YOLO-formatted targets. |
| `inference.py` | Standalone entry point for running YOLOv1 on sample images using the settings in `config_yolov1.yaml`. |
| `loss.py` | Implementation of the YOLOv1 loss function (`YOLOLoss`) with a compatibility alias `YOLOv1Loss`. |
| `train.py` | End-to-end training script that wires together the dataset, model, loss, optimisation, evaluation, and checkpointing flows. |
| `training_utils.py` | Helper routines for calculating mAP, early stopping, and optional visualisation during training. |
| `utils.py` | Shared utilities for loading Darknet weights, decoding predictions, performing NMS, and computing IoU. |
| `yolov1.py` | The YOLOv1 neural network architecture definition. |
| `TRAINING.md` | Background reading and practical tips for training and tuning YOLOv1 (complements this README). |

## Running YOLOv1

### Prerequisites

- Python 3.9+
- PyTorch and torchvision (GPU acceleration recommended)
- `pycocotools`, `tqdm`, `matplotlib`, and other dependencies listed in the project root
- COCO-format annotations plus image assets, referenced by `config_yolov1.yaml`

### Training

1. Update `config_yolov1.yaml` with dataset paths, hyperparameters, and logging directories.
2. (Optional) Place pretrained Darknet weights at the `model.weights_path` location specified in the config.
3. Run the training script from the repo root:
   ```bash
   python -m pytorch.models.yolo.train
   ```
   - Checkpoints and logs are emitted under `logging.log_dir` with timestamped subdirectories.
   - Early stopping and learning-rate scheduling are driven by the config values.

### Inference

1. Confirm the `config_yolov1.yaml` inference section points to the desired weights and thresholds.
2. Execute the inference demo:
   ```bash
   python -m pytorch.models.yolo.inference
   ```
3. The script annotates the example image defined inside `inference.py` and stores the result under `../../sample_output/` relative to the module.

To integrate YOLOv1 with the shared service pipeline, point `pytorch/config.yaml` (or task-specific configs) to the desired model and weight paths, then use `python pytorch/main.py`.

## Testing & Validation

| Scope | Recommended Checks |
| --- | --- |
| Unit smoke tests | Run existing `tests/test_infer.py` to ensure the inference helpers still return annotated images: `pytest tests/test_infer.py`. |
| Model forward pass | From an interactive shell, instantiate `YOLOv1` and confirm a dummy batch produces the expected output shape `(B, S, S, C + B*5)`. |
| Loss stability | Feed synthetic predictions/targets through `YOLOLoss` to verify finite outputs and gradient backprop (e.g., with a small random tensor in a notebook). |
| mAP evaluation | After training, run the validation step in `train.py`; verify the logged mAP and inspect saved visualisations for qualitative quality. |
| Integration | Execute `python pytorch/main.py` with a YOLO configuration to ensure the end-to-end infer pipeline draws detections and metadata overlays without errors. |

For comprehensive regression coverage, pair these checks with dataset-specific acceptance criteria (e.g., minimum mAP, inference FPS) relevant to your deployment.

