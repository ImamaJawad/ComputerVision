# Image Classification using Hand-crafted Features, Neural Networks, and CNNs

**Module:** ECS8053 – Computer Vision (Spring 2025)

**Author:** Imama Jawad  
**Student ID:** 40462364  
**Institution:** Queen’s University Belfast  
**Date:** February 3, 2025

---

## Project overview

This repository contains code, experiments, and results for the project *Image Classification using Hand-crafted Features, Neural Networks, and CNNs* (ECS8053). The goal is to compare three families of approaches on the same image dataset:

1. Hand-crafted feature pipelines (SIFT / ORB + Fisher Vector + classifier)
2. Simple feed-forward Neural Network (features from raw images / engineered features)
3. Convolutional Neural Network (end-to-end training; transfer learning & custom CNNs)

The repository includes dataset-preprocessing, training / evaluation scripts, notebooks used for analysis, and the Google Colab appendix used in coursework.

---

## Repository structure

```
README.md
requirements.txt
environment.yml  # optional
LICENSE
data/
  ├─ raw/           # original dataset (not included in repo) — see DATA.md
  ├─ processed/     # processed / resized images used in experiments
notebooks/
  ├─ 01_data_exploration.ipynb
  ├─ 02_handcrafted_features.ipynb
  ├─ 03_simple_nn.ipynb
  └─ 04_cnn_experiments.ipynb
src/
  ├─ data.py            # dataset loader, transforms, augmentation
  ├─ features.py        # SIFT, ORB extraction, BoW, Fisher Vector utilities
  ├─ models/
  │   ├─ simple_nn.py
  │   ├─ cnn.py
  │   └─ torch_utils.py
  ├─ train.py           # general training loop wrapper
  ├─ evaluate.py        # evaluation metrics + confusion matrix + plotting
  └─ utils.py           # helpers
reports/
  ├─ figures/           # saved plots (accuracy curves, confusion matrices)
  └─ final_report.pdf   # (optional) final report
colab/                  # Google Colab notebook and instructions (Appendix A)
  └─ ecs8053_colab.ipynb

```

---

## Getting started

### Prerequisites

This project runs on Python 3.8+ and uses common ML libraries. Two installation options are provided.

1. **pip**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

2. **conda (optional)**

```bash
conda env create -f environment.yml
conda activate ecs8053
```

**Note:** Large libraries such as `opencv-contrib-python`, `torch`, and `tensorflow` are optional depending on which experiments you plan to run. See `requirements.txt`.

### requirements.txt (example)

```
numpy
scipy
scikit-learn
opencv-contrib-python
matplotlib
pandas
tqdm
scikit-image
torch
torchvision
tensorboard
joblib
faiss-cpu    # optional (for large-scale feature quantisation)
vlfeat or opensswift bindings (if using Fisher vector from other libs)
```

> If you will run CNN training on GPU, install the PyTorch build appropriate to your CUDA version following the official instructions.

---

## Data

The raw dataset is *not* included in the repository for size and licensing reasons. Place the dataset in `data/raw/` following the structure described in `DATA.md` (example: `data/raw/train/<class>/*.jpg`, `data/raw/test/<class>/*.jpg`).

The notebooks and `src/data.py` include utilities to resize, normalise, and split the data into `data/processed/` for experiments.

**Important:** If you used a public dataset (e.g., CIFAR, MNIST, or a custom course dataset), include the dataset citation and a download script in `DATA.md`.

---

## How to run experiments

Below are minimal examples for running the main experiment types. All scripts accept `--config` or CLI flags; see `--help` for details.

### 1) Hand-crafted features pipeline (SIFT / ORB + Fisher Vector + classifier)

```bash
python src/features.py \
  --data-dir data/processed/train \
  --method sift \
  --vocab-size 64 \
  --output features/sift_fv_train.pkl

python src/train.py \
  --features features/sift_fv_train.pkl \
  --model svc \
  --out models/sift_fv_svc.joblib

python src/evaluate.py --model models/sift_fv_svc.joblib --data data/processed/test
```

Notes:
- `src/features.py` supports `sift` and `orb` extraction, Bag-of-Words, and Fisher Vector encoding.
- For Fisher Vectors we used `scikit-learn`'s GMM + custom encoder (or an external library). Comments in the code point to references used.

### 2) Simple feed-forward neural network

```bash
python src/train.py --config configs/simple_nn.yaml
```

This trains a small MLP on flattened/resized images or precomputed features. The YAML config contains hyperparameters (learning rate, batch size, epochs).

### 3) Convolutional Neural Network (transfer learning / custom CNN)

```bash
python src/train.py --config configs/cnn_resnet_transfer.yaml
```

- The repo includes both a small custom CNN and examples using transfer learning (ResNet18 / MobileNetV2) from `torchvision.models`.
- TensorBoard logs are written to `runs/` and model checkpoints to `checkpoints/`.

---

## Notebooks

If you prefer an interactive route, run the notebooks in `notebooks/` or open the Google Colab at `colab/ecs8053_colab.ipynb`.

Notebooks:
- `01_data_exploration.ipynb` — dataset statistics, sample images, class balance
- `02_handc
