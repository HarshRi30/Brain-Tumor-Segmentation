"""
config.py — Central configuration for BraTS 2023 GLI Brain Tumor Segmentation
All notebooks import from here. Only change DATASET_PATH to match your DGX setup.
"""

import os
from pathlib import Path

# =============================================================================
# ⚙️  ONE LINE TO CHANGE ON DGX — set your actual dataset path here
# =============================================================================
DATASET_PATH = "/home/yourname/BraTS2023_Training_Data"   # ← CHANGE THIS on DGX
VALIDATION_PATH = "/home/yourname/BraTS2023_Validation_Data"  # ← CHANGE THIS on DGX
PREPROCESSED_PATH = "/home/yourname/BraTS2023_Preprocessed"   # ← CHANGE THIS on DGX

# Output paths (auto-created)
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"
LOGS_DIR       = "./logs"

# =============================================================================
# 📂 BraTS 2023 GLI File Naming Convention
# =============================================================================
# Pattern: BraTS-GLI-{XXXXX}-{YYY}-{modality}.nii.gz
MODALITIES = {
    "flair": "t2f",   # FLAIR
    "t1":    "t1n",   # T1 native
    "t1ce":  "t1c",   # T1 contrast-enhanced
    "t2":    "t2w",   # T2
    "seg":   "seg",   # Ground truth segmentation
}

# =============================================================================
# 🏷️  BraTS 2023 Segmentation Labels
# =============================================================================
# Label 0 → Background
# Label 1 → Necrotic Tumor Core (NCR)
# Label 2 → Peritumoral Edema (ED)
# Label 3 → Enhancing Tumor (ET)      ← was "4" in BraTS 2021
LABEL_NAMES = {0: "Background", 1: "NCR", 2: "ED", 3: "ET"}
NUM_CLASSES = 4  # including background

# BraTS evaluation sub-regions (derived from labels):
#  WT (Whole Tumor)  = labels 1 + 2 + 3
#  TC (Tumor Core)   = labels 1 + 3
#  ET (Enhancing)    = label  3
REGION_LABELS = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}

# =============================================================================
# 🧠 Model & Training Hyperparameters (tuned for 1 GPU on DGX)
# =============================================================================
# -- Data split --
TRAIN_RATIO       = 0.80   # 80% training
VAL_RATIO         = 0.10   # 10% validation
TEST_RATIO        = 0.10   # 10% test
MAX_PATIENTS      = 600    # Use subset — increase if GPU VRAM allows

# -- Patch-based training --
PATCH_SIZE        = (96, 96, 96)    # 96³ patches — fits comfortably on 1 GPU
PATCH_OVERLAP     = (16, 16, 16)    # Overlap for sliding window inference
NUM_PATCHES_TRAIN = 2               # Patches per volume per epoch

# -- Network --
IN_CHANNELS       = 4               # FLAIR, T1, T1ce, T2
OUT_CHANNELS      = 4               # 4 classes (BG + 3 tumor regions)
INIT_FEATURES     = 32              # Starting feature maps in U-Net encoder

# -- Training --
BATCH_SIZE        = 2
NUM_EPOCHS        = 150
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-5
LR_PATIENCE       = 10             # ReduceLROnPlateau patience
LR_FACTOR         = 0.5
EARLY_STOP_PATIENCE = 25
NUM_WORKERS       = 4
PIN_MEMORY        = True

# -- K-Fold --
N_FOLDS           = 3
FOLD              = 0               # Which fold to train (0, 1, or 2)

# -- Loss --
LOSS_DICE_WEIGHT  = 0.5
LOSS_CE_WEIGHT    = 0.5

# -- Seeds --
RANDOM_SEED       = 42

# -- Quick Test Mode (run on tiny subset to verify pipeline) --
QUICK_TEST        = False           # Set True to run on 10 patients only
QUICK_TEST_N      = 10

# =============================================================================
# 💾 Device
# =============================================================================
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 📁 Auto-create output directories
# =============================================================================
for _dir in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, PREPROCESSED_PATH]:
    os.makedirs(_dir, exist_ok=True)
