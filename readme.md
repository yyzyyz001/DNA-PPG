# DNA-PPG: A Foundation Model for Photoplethysmography via Dual Neighborhood Alignment

> **Note:** This repository contains the official PyTorch implementation of the paper **"DNA-PPG: A Foundation Model for Photoplethysmography via Dual Neighborhood Alignment"** submitted to **IJCAI 2026**.

## 🚀 Introduction

**DNA-PPG** is a novel pre-training framework designed to learn robust and universal representations for Photoplethysmography (PPG) signals. It addresses the limitations of existing physiological foundation models—specifically the manifold distortion caused by rigid hard-negative sampling and the precision loss from coarse discretization.

Our framework introduces **Dual Neighborhood Alignment**:

1.  **Morphology-Aware Self-Supervised Branch (Morph-SSL):** Uses Time-Frequency Soft Weighting (TF-Soft) to capture universal signal dynamics.
2.  **Physiological Semantic Alignment Branch (Phys-Align):** Projects physiological indicators into a continuous semantic space to embed precise physiological priors.

Pre-trained on **10.7 million PPG segments** from over 8,400 subjects, DNA-PPG achieves state-of-the-art performance on downstream regression and classification tasks.

## 📂 Repository Structure

```text
.
├── baselines/            # Implementations of comparison methods 
├── ckpt/                 # Directory for saving model checkpoints
├── downstream/           # Code for downstream tasks
├── models/               # Model definitions
├── preprocessing/        # General preprocessing utilities
├── augmentations.py      # Signal augmentations
├── dataset.py            # PyTorch Dataset classes for data loading
├── dsPreProcess.py       # Preprocessing scripts specific to downstream datasets
├── losses.py             # Implementation of Morphology-Aware Loss (L_Morph) and Phys-Align Loss (L_Phys)
├── preProcessMesa.py     # Data cleaning and segmentation for MESA dataset
├── preProcessVital.py    # Data cleaning and segmentation for VitalDB dataset
├── train.py              # Main pre-training script
├── utilities.py          # General helper functions
└── README.md             # This file
```

## 🛠️ Environment Requirements

To reproduce the experiments, please set up the environment using the following dependencies:

- Python: 3.10
- PyTorch: 2.3.1
- CUDA: 12.1
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Tqdm

```bash
pip install torch numpy scipy pandas scikit-learn tqdm
```

## 📊 Data Preparation

Due to privacy regulations and licensing agreements, we cannot provide the raw datasets directly. Please request access from the official sources:

### Pre-training Datasets

1. **VitalDB:** https://physionet.org/content/vitaldb/
2. **MESA:** https://sleepdata.org/datasets/mesa

After downloading, place the raw data in a local directory and run the specific preprocessing scripts to generate 10-second segments and align physiological labels:

```bash
# Preprocess VitalDB
python preProcessVital.py

# Preprocess MESA
python preProcessMesa.py
```

### Downstream Datasets

Refer to `dsPreProcess.py` for handling downstream datasets (PPG-BP, PPG-DaLiA, WESAD, etc.).

## 🚀 Usage

### 1. Pre-training

To pre-train the DNA-PPG model using the Dual Neighborhood Alignment strategy:

```bash
python train.py --epoch 10 --model "resnet1d" --transform_ssl --transform_sup --use-tfc --data-source "all"
```

### 2. Downstream Evaluation

To fine-tune the pre-trained encoder on specific tasks:

```bash
python dsPreProcess.py
python -m downstream.parallel_executor
cd downstream/
python outcome_regression_all.py --model ../ckpt/dna_ppg.pt
python outcome_classification_all.py --model ../ckpt/dna_ppg.pt
```

### 3. Loading the Pre-trained Model

To load the pre-trained DNA-PPG checkpoint for your own research, ensure you initialize the `ResNet1D` backbone with the configuration used during pre-training:

```python
import torch
from models.resnet1d import ResNet1D
from utilities import load_model

# 1. Initialize the backbone with pre-training configuration
model = ResNet1D(
    in_channels=1,
    base_filters=64,
    kernel_size=3,
    stride=2,
    groups=1,
    n_block=18,
    n_classes=512,
)

# 2. Load the checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model, "./ckpt/dna_ppg.pt", device)
model.to(device)
model.eval()

print("DNA-PPG model loaded successfully.")
```

## 🧩 Model Architecture, Losses & Checkpoints

The core logic and pre-trained resources are organized as follows:

- **`ckpt/dna_ppg.pt`**:
   The **official pre-trained checkpoint** of DNA-PPG.
  - It contains the weights of the **ResNet-1D** encoder (approx. 4.99M parameters).
  - Trained on the full dataset (VitalDB + MESA, 10.7M segments) using the optimal joint strategy (α=0.7) as reported in the paper.
  - This file should be used to initialize the backbone for all downstream evaluations.
- **`models/`**:
   Contains the definition of the ResNet-1D backbone.
- **`losses.py`**:
  - `loss_morph`: Implements **Eq. (5)** from the paper, utilizing the soft weights $w_{ij}$ to preserve morphology-invariant neighborhoods.
  - `loss_phys`: Implements **Eq. (11)**, calculating the affinity matrix $A$ based on continuous physiological semantic distances.

## 📝 Citation

If you find this code useful, please cite our paper (BibTeX will be updated after the review process):

```bibtex
@article{DNA-PPG2026,
  title={DNA-PPG: A Foundation Model for Photoplethysmography via Dual Neighborhood Alignment},
  author={Anonymous Authors},
  journal={Submitted to IJCAI},
  year={2026}
}
```

## 🙏 Acknowledgements

We thank the authors of the following open-source projects, whose codebases provided valuable foundations for this work:

- **PaPaGei:** https://github.com/Nokia-Bell-Labs/papagei-foundation-model
- **TF-C:** https://github.com/mims-harvard/TFC-pretraining

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for details.