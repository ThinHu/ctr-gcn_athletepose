# AthletePose3D Action Recognition with CTR-GCN

A PyTorch-based pipeline for Human Action Recognition (HAR) using a Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN) trained on the AthletePose3D dataset.

## Key Features
* **COCO 17-Joint Graph Adaptation:** Custom spatial graph built specifically for 17-joint, 2D coordinate skeletons, replacing the default 25-joint NTU-RGB+D graph.
* **Robust Data Pipeline:** Custom PyTorch Dataset featuring on-the-fly NaN/inf imputation, zero-centering normalization (relative to the root joint), and dynamic directory crawling using Regex.
* **Heavy Regularization:** Combats coordinate-data overfitting using Spatial Gaussian Noise, Temporal Frame Sampling/Padding, L2 Weight Decay (0.0005), and Dropout (0.5).
* **Optimized Training:** Implements Learning Rate Warmup, Step Decay, gradient clipping, and Early Stopping based on validation accuracy.
* **Deep Evaluation:** Automatically generates normalized confusion matrices and detailed classification reports (Precision, Recall, F1-Score).

## Directory
```
ctr-gcn_athletepose/
├── data/                   # Data directory (often ignored in Git)
│   ├── train/              
│   ├── valid/              
│   └── test/               
├── notebooks/              # Jupyter notebooks for exploration & visualization
│   └── 01_training_experiment.ipynb
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── data/               # Data loaders and dataset classes
│   │   ├── __init__.py
│   │   └── dataset.py      
│   ├── models/             # Neural network architectures and graph definitions
│   │   ├── __init__.py
│   │   ├── graph.py        
│   │   └── ctrgcn.py       
│   ├── engine/             # Training and evaluation loops
│   │   ├── __init__.py
│   │   └── trainer.py      
│   └── utils/              # Helper functions, metrics, and visualization
│       ├── __init__.py
│       ├── metrics.py      
│       └── seed.py         
├── .gitignore              # Files to ignore (e.g., /data, __pycache__, .pth)
├── requirements.txt        # Project dependencies
├── main.py                 # Main entry point to run the pipeline
└── README.md               # Project documentation
```

## Pipeline Overview
1. **Graph Setup:** Maps the physical inward/outward connections of the human body (GraphCOCO).
2. **Model Initialization:** Initializes the 10-layer CTR-GCN architecture, patched to handle 2-channel (X, Y) input without dimensionality crashes.
3. **Training Loop:** Processes batches (size 32) using SGD with Nesterov momentum, profiling data-loading vs. network-compute bottlenecks, and saving the best weights (best_ctrgcn.pth).
4. **Inference & Metrics:** Evaluates the optimal model against the validation set, plotting a heatmap of misclassifications to identify overlapping athletic movements.

## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run training:
   `python main.py --data_path /path/to/athletepose3d --batch_size 16`## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run training:
   `python main.py --data_path /path/to/athletepose3d --batch_size 16`vv

## Acknowledgements

This project builds upon the foundational work and codebase of the CTR-GCN framework and AthletePose3D Dataset upon here:

* [Uason-Chen/CTR-GCN](https://github.com/Uason-Chen/CTR-GCN.git)
* [calvinyeungck/AthletePose3D](https://github.com/calvinyeungck/AthletePose3D.git)
