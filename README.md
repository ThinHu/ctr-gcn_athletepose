# AthletePose3D Action Recognition using CTR-GCN

This repository/notebook contains a complete **PyTorch pipeline** for training a **Channel-wise Topology Refinement Graph Convolution Network (CTR-GCN)** on the **AthletePose3D dataset**.

It is designed to run efficiently in a **Kaggle Notebook environment** (Dual T4 or P100 GPUs) and handles dynamic dataset loading, data augmentation, and robust training regularization.

---

# Project Overview

Skeleton-based human action recognition requires modeling both the **spatial configurations of human joints** and their **temporal dynamics across frames**.

This project utilizes **Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN)**, which dynamically learns different graph topologies for different channels to better capture complex athletic movements.

---

# Key Modifications & Features

### COCO 17-Joint Adaptation
Replaced the default **NTU-RGB+D 25-joint graph** with a **standard 17-joint COCO skeleton graph**.

### 2D Coordinate Handling
Configured the network to accept **2D inputs**

Also patched internal CTR-GCN **channel-reduction logic** to prevent **zero-channel tensor crashes**.

### Dynamic Directory Crawling
Custom PyTorch Dataset that:

- Recursively crawls `train`, `test`, and `valid` folders
- Extracts **action labels automatically from filenames using Regex**

### Aggressive Regularization

To combat **overfitting on coordinate data**:

#### Spatial Augmentation
- Gaussian noise injection

#### Temporal Augmentation
- Random frame sampling
- Dynamic edge padding

#### Network Regularization
- 50% Dropout
- L2 Weight Decay = `0.0005`

### Automated Training Loop

Includes:

- Learning Rate Warmup
- Step Decay
- Gradient Clipping
- Early Stopping (based on validation accuracy)

---

# 📂 Dataset Structure

The DataLoader expects the **AthletePose3D dataset** extracted as `.npy` arrays in the following structure:

```plaintext
athletepose3d/
├── train/
│   ├── S1/
│   │   ├── Axel_1_cam_1_coco.npy
│   │   ├── Axel_1_cam_2_coco.npy
│   │   └── ...
│   └── S2/
├── test/
└── valid/
```

**Note**

The `train` directory is used for traning.

The `valid` directory is used for **strict model evaluation**.

---

# ⚙️ Environment & Setup

**Platform:** Kaggle Notebooks  
**Accelerator:** GPU (T4×2 or P100)

The project runs using the **standard Kaggle PyTorch environment**.

### Required Libraries

```python
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
```

---

# 🚀 Usage Guide (Notebook Execution)

The project is divided into sequential executable notebook blocks.

---

## Graph Definition

Defines the **GraphCOCO class**, establishing the **inward/outward physical connections** of the **17-joint human skeleton**.

---

## CTR-GCN Architecture

Contains the core spatial-temporal blocks with channel-wise refinemnet technique:

- `TemporalConv`
- `MultiScale_TemporalConv`
- `CTRGC`
- `unit_tcn`
- `TCN_GCN_unit`
- `CTRGCN_Model`

and the final **classification model**.

---

## Dataset & DataLoader

Contains the `AthletePose3DDataset` class.

Responsibilities include:

- NaN cleaning
- Zero-centering normalization (relative to the root joint)
- Temporal normalization to **exactly 64 frames**

---

## Training Loop

Executes the full training process.

### Default Parameters

| Parameter | Value |
|-----------|------|
| Epochs | 100 |
| Batch Size | 32 |
| Base Learning Rate | 0.05 |
| Early Stop Patience | 20 |

When a **new highest validation accuracy** is reached, the model is saved.

---

# Evaluation & Metrics

The final evaluation cell generates:

### Normalized Confusion Matrix

Visualizes which athletic movements the model struggles to differentiate.

Displayed using a **Seaborn heatmap**.

### Classification Report

Outputs:

- Precision
- Recall
- F1-score

for **all 18 target classes**.


# Acknowledgments

### CTR-GCN
Based on the paper:

**"Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition"**  
(ICCV 2021)

### AthletePose3D Dataset
For providing the comprehensive **multi-view sports action dataset**.
