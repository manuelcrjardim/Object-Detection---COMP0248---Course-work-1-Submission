# Multi-Task Hand Gesture Recognition

A multi-task deep learning model that performs simultaneous hand gesture classification, bounding box detection, and segmentation mask prediction from RGB-D (colour + depth) image pairs.

## Overview

The model is trained on a dataset of 10 hand gesture classes captured with paired RGB and depth cameras. Each frame has three ground truth labels: a gesture class, a bounding box, and a pixel-level segmentation mask. The network learns all three tasks jointly using a weighted composite loss.

**Gesture classes:** G01_call, G02_dislike, G03_like, G04_ok, G05_one, G06_palm, G07_peace, G08_rock, G09_stop, G10_three

## Architecture

`MultiTaskRGBDNet` is a custom encoder-decoder network with early fusion of RGB and depth streams.

- **RGB stem:** 3-channel input -> 32 feature maps
- **Depth stem:** 1-channel input -> 16 feature maps
- **Encoder:** 4 residual blocks (64 -> 128 -> 256 -> 512 channels) with Squeeze-and-Excitation (SE) attention
- **Decoder:** 4 upsampling stages with skip connections, producing a full-resolution segmentation mask
- **Classification head:** Global average pool on bottleneck -> 2-layer MLP
- **Bounding box head:** 5x5 spatial pool on bottleneck -> 3-layer MLP with Sigmoid output (normalised XYXY coordinates)

## Project Structure

```
.
├── model.py          # Network definition (MultiTaskRGBDNet, ResBlock, SEBlock)
├── dataloader.py     # Dataset class with in-memory pre-loading and augmentation
├── utils.py          # Augmentation pipeline, IoU/Dice metrics, test set parsing
├── train.py          # Training loop and hyperparameter grid search
├── evaluate.py       # Inference and metric reporting on the test set
└── visualise.py      # Confusion matrix and prediction visualisations
```

## Training

Training uses a weighted composite loss over the three tasks:

| Loss component | Criterion          | Weight |
|----------------|--------------------|--------|
| Classification | CrossEntropyLoss   | 1.0    |
| Bounding box   | SmoothL1Loss       | 15.0   |
| Segmentation   | BCEWithLogitsLoss  | 2.0    |

The training script runs a grid search over activation functions (`ReLU`, `LeakyReLU`) and learning rates (`1e-3`, `1e-4`) using the Adam optimiser. Early stopping is applied with a configurable patience window (`epoch_margin`).

To run training:

```bash
python train.py
```

Model weights for each configuration are saved as `model_{activation}_{lr}.pth`. The best model by macro F1 is saved separately as `best_model_gridsearch.pth`. All experiment metrics and loss histories are written to `all_gridsearch_metrics.json`.

## Data Augmentation

Each training sample is augmented `num_augmentations` times (default: 4) using:

- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±15 degrees)
- Random affine (translation up to 10%, scale 0.9–1.1)

Augmentations are applied consistently across the RGB image, depth map, segmentation mask, and bounding box using `torchvision.transforms.v2`.

## Dataset Format

The training set is expected at `dataset/RGB_depth_annotations/` with the structure:

```
RGB_depth_annotations/
└── <student_id>/
    └── <folder>/
        └── <gesture_label>/
            └── <clip>/
                ├── rgb/         frame_*.png
                ├── depth/       frame_*.png
                ├── depth_raw/   frame_*.npy
                └── annotation/  frame_*.png
```

The test set is expected at `dataset/test_set/` with gesture directories directly inside it.

## Evaluation

```bash
python evaluate.py
```

Reported metrics:

- Classification accuracy and macro F1 score
- Mean bounding box IoU and detection accuracy at IoU threshold 0.5
- Mean segmentation IoU and Dice coefficient

## Visualisation

```bash
python visualise.py
```

Produces:

- `confusion_matrix.png` — heatmap of predicted vs true gesture classes
- `visualization_samples.png` — side-by-side ground truth and prediction overlays for sample frames

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pillow
- scikit-learn
- matplotlib
- seaborn
- tqdm