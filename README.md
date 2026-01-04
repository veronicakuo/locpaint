# LocPaint: Context Encoder with Localizer Discriminator

A PyTorch implementation of GAN-based image inpainting algorithm with a novel **localizer discriminator** paradigm. I have written a report for more context [here](https://drive.google.com/file/d/1tKQ0nT-1N6a2r1LytKhrtLatNP27wnIs/view?usp=sharing).
Thank you to Professor Olga Russakovsky and Professor Vikram Ramaswamy for their support and guidance to this project!

## Overview

**Image inpainting** is the task of filling in missing pixels in an image, with applications including image recovery, object removal, and image editing.

[Pathak et al. (2016)](https://arxiv.org/abs/1604.07379) pioneered the first major work in deep learning-based image inpainting, featuring a GAN-style architecture with a context encoder adversarially supervised by a discriminator. Since then, many other approaches have been proposed, including more recent diffusion-based methods ([Lugmayr et al., 2022](https://arxiv.org/abs/2201.09865)).

### Key Intuition

To my knowledge, all previous GAN inpainting approaches leverage a **binary classifier** which classifies generator outputs as "real" or "fake." Yet image inpainting lends itself well to discriminators that are **localizers**: if someone thinks an inpainted image looks "fake," they should be able to identify **where** the "fake" region is.

### Why localization?

I thought a localizer discriminator would work well for inpainting for the following reasons:

1. **Feasibility.** Square masks used to define "missing" pixels provide a natural bounding box ground truth for the localizer. This is a self-supervised paradigm that eliminates the need for extra annotations.

2. **Specificity.** The generator receives more specific gradient signals from a localizer discriminator. A binary "real/fake" prediction doesn't tell the generator *which parts* of the image are reconstructed well or poorly, but a localizer discriminator enables this spatial feedback.

3. **Reiteration.** When an image is reconstructed, some parts may look real while others look fake. A localizer discriminator could identify fake-looking regions and have the generator refine them iteratively, creating progressively realistic reconstructions.

### Contribution

Building on [Pathak et al. (2016)](https://arxiv.org/abs/1604.07379), I implement an inpainting model with a discriminator that acts as a **localizer** instead of a classifier. I call this approach **LocPaint** and compare it against:
- **L2-only baseline** (reconstruction loss only, no adversarial training)
- **Pathak et al. baseline** (binary classifier discriminator)
- **Phased LocPaint** (staged training: binary → localizer)

### Key Features

- **Baseline Implementation**: Reproduction of Pathak et al.'s Context Encoder
- **Localizer Discriminator**: Novel discriminator with confidence + bounding box prediction heads
- **Phased Training**: Staged approach that transitions from binary to localizer discriminator
- **Comprehensive Evaluation**: MSE, PSNR, SSIM, and LPIPS metrics
- **Reproducible Splits**: Fixed train/val/test splits with configurable seeds

## Quick Start (Google Colab)

### 1. Setup

```python
# Check GPU
!nvidia-smi -L
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Install dependencies
!pip install -q torch torchvision matplotlib opencv-python-headless pyyaml tensorboard tqdm scikit-image lpips

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/research/locpaint
```

### 2. Project Structure

```
locpaint/
├── baseline/
│   ├── checkpoints_l2/       # L2-only model (no adversarial)
│   ├── checkpoints_pathak/   # Pathak et al. reproduction
│   ├── baseline_context_encoder.py
│   └── train_baseline.py
├── localizer/
│   ├── checkpoints/          # Localizer model
│   ├── context_encoder.py
│   └── train_localizer.py
├── staged/
│   ├── checkpoints/          # Phased model
│   ├── staged_context_encoder.py
│   └── train_staged.py
├── evaluation/               # Evaluation results
├── comparison_results/       # Visual comparisons
├── utils/
│   ├── __init__.py
│   ├── data_split.py
│   ├── metrics.py
│   └── visualization.py
├── evaluate_models.py
├── compare_models.py
├── test_discriminator_detection.py
└── context_encoder_localizer_demo.ipynb
```

## Training

### 1. L2-Only Baseline (No Adversarial Loss)

```bash
!python baseline/train_baseline.py \
    --data_root ./data \
    --dataset cifar10 \
    --image_size 128 \
    --mask_size 64 \
    --batch_size 64 \
    --epochs 30 \
    --lambda_rec 1.0 \
    --lambda_adv 0.0 \
    --checkpoint_dir ./baseline/checkpoints_l2
```

### 2. Pathak et al. Baseline

```bash
!python baseline/train_baseline.py \
    --data_root ./data \
    --dataset cifar10 \
    --image_size 128 \
    --mask_size 64 \
    --batch_size 64 \
    --epochs 30 \
    --lambda_rec 0.999 \
    --lambda_adv 0.001 \
    --checkpoint_dir ./baseline/checkpoints_pathak
```

### 3. Localizer (Ours)

```bash
!python localizer/train_localizer.py \
    --data_root ./data \
    --dataset cifar10 \
    --image_size 128 \
    --mask_size 64 \
    --batch_size 64 \
    --epochs 30 \
    --lambda_rec 0.999 \
    --lambda_conf 0.0005 \
    --lambda_loc 0.0 \
    --lambda_iou 0.0005 \
    --d_capacity_factor 0.5 \
    --d_update_ratio 2 \
    --checkpoint_dir ./localizer/checkpoints
```

### 4. Phased Localizer (Ours)

```bash
!python staged/train_staged.py \
    --data_root ./data \
    --dataset cifar10 \
    --image_size 128 \
    --mask_size 64 \
    --batch_size 64 \
    --epochs 30 \
    --transition_epoch 20 \
    --warmup_epochs 5 \
    --checkpoint_dir ./staged/checkpoints
```

## Evaluation

### Quantitative Evaluation

```bash
!python evaluate_models.py \
    --l2_checkpoint ./baseline/checkpoints_l2/best_model.pth \
    --baseline_checkpoint ./baseline/checkpoints_pathak/best_model.pth \
    --localizer_checkpoint ./localizer/checkpoints/best_model.pth \
    --staged_checkpoint ./staged/checkpoints/best_model.pth \
    --dataset cifar10 \
    --eval_set test \
    --output_dir ./evaluation
```

**Output:**
- `evaluation_results.json` - All metrics
- `evaluation_table.tex` - LaTeX table (bold best, underlined second best)

### Visual Comparison

```bash
!python compare_models.py \
    --l2_checkpoint ./baseline/checkpoints_l2/best_model.pth \
    --baseline_checkpoint ./baseline/checkpoints_pathak/best_model.pth \
    --localizer_checkpoint ./localizer/checkpoints/best_model.pth \
    --staged_checkpoint ./staged/checkpoints/best_model.pth \
    --output_dir ./comparison_results \
    --mask_size 64 \
    --image_seed 42
```

**Output:**
- `model_comparison.png` - Side-by-side comparison (10 images, one per CIFAR-10 class)
- `model_comparison_zoomed.png` - Zoomed into inpainted regions

### Discriminator Diagnostics

Test what the discriminator is detecting (boundaries vs. content/texture):

```bash
# For localizer model
!python test_discriminator_detection.py \
    --checkpoint ./localizer/checkpoints/best_model.pth \
    --model_type localizer \
    --output_dir ./diagnostic_localizer

# For staged model (tests BOTH discriminators)
!python test_discriminator_detection.py \
    --checkpoint ./staged/checkpoints/best_model.pth \
    --model_type staged \
    --output_dir ./diagnostic_staged
```

**Diagnostic Tests:**
| Test | Description | What High IoU Means |
|------|-------------|---------------------|
| Real images | Unmodified images | D falsely detecting regions |
| Gaussian blur | Blurred region, NO boundary | D detecting texture, not boundary |
| Noise regions | Noisy region, NO boundary | D detecting noise artifacts |
| Mean color | Uniform patch, NO boundary | D detecting color mismatch |
| Completed | Inpainted with boundary | D detecting boundary (expected) |
| Fully reconstructed | Entire image through G | D detecting generator artifacts |

## Loss Functions

### Discriminator Loss

```
L_D = L_real_conf + L_fake_conf + λ_loc × L_fake_loc + λ_iou × L_fake_iou
```

| Term | Formula | Purpose |
|------|---------|---------|
| L_real_conf | BCE(D_conf(x_real), 0) | Real images → low confidence |
| L_fake_conf | BCE(D_conf(x_fake), 1) | Fake images → high confidence |
| L_fake_loc | SmoothL1(bbox_pred, bbox_true) | Accurate bbox prediction |
| L_fake_iou | 1 - IoU(bbox_pred, bbox_true) | Maximize bbox overlap |

### Generator Loss

```
L_G = λ_rec × L_rec + λ_conf × L_conf + λ_loc × L_loc + λ_iou × L_iou
```

| Term | Formula | Purpose |
|------|---------|---------|
| L_rec | MSE(x, x̂) | Reconstruction quality |
| L_conf | BCE(D_conf(x̂), 0) | Fool D (low confidence) |
| L_loc | -SmoothL1(bbox_pred, bbox_true) | Maximize D's bbox error |
| L_iou | IoU(bbox_pred, bbox_true) | Penalize when D finds region |

## Metrics

| Metric | Measures | Range | Better |
|--------|----------|-------|--------|
| **MSE** | Pixel error | [0, ∞) | Lower ↓ |
| **PSNR** | Log pixel error (dB) | [0, ∞) | Higher ↑ |
| **SSIM** | Structural similarity | [0, 1] | Higher ↑ |
| **LPIPS** | Perceptual distance (AlexNet) | [0, 1] | Lower ↓ |

## Checkpoints

Each training run saves:

| Checkpoint | Criterion |
|------------|-----------|
| `best_model.pth` | Lowest training reconstruction loss |
| `best_psnr_model.pth` | Highest validation PSNR |
| `best_ssim_model.pth` | Highest validation SSIM |
| `best_lpips_model.pth` | Lowest validation LPIPS |
| `checkpoint_epoch_N.pth` | Periodic saves |

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
pillow
tqdm
lpips
scikit-image
opencv-python-headless
pyyaml
tensorboard
pytorch-msssim  # Optional, for faster SSIM
```

Install all dependencies:
```bash
pip install torch torchvision matplotlib opencv-python-headless pyyaml tensorboard tqdm scikit-image lpips
```

## Citation

If you use this code, please cite:

```bibtex
@article{locpaint2026,
  title={Localization-Guided Adversarial Inpainting},
  author={Veronica Kuo},
  year={2024}
}
```

And the original Context Encoder paper:

```bibtex
@inproceedings{pathak2016context,
  title={Context Encoders: Feature Learning by Inpainting},
  author={Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A},
  booktitle={CVPR},
  year={2016}
}
```

## License

MIT License

## Acknowledgments

- [Context Encoders](https://github.com/pathak22/context-encoder) by Pathak et al.
