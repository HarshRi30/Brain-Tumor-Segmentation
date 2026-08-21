"""
metrics.py — All 5 evaluation metrics for BraTS 2023 GLI.

Metrics computed per BraTS sub-region (WT, TC, ET):
  1. Dice Score
  2. IoU (Jaccard Index)
  3. HD95 (95th percentile Hausdorff Distance)
  4. Sensitivity (Recall)
  5. Specificity
"""

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Region derivation from BraTS 2023 label map
# ---------------------------------------------------------------------------

def get_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert BraTS 2023 integer label map to 3 binary sub-region masks.

    Labels:  0=BG, 1=NCR, 2=ED, 3=ET
    Regions: WT = 1+2+3,  TC = 1+3,  ET = 3

    Args:
        seg : integer array (H, W, D)
    Returns:
        dict with keys 'WT', 'TC', 'ET' — each a binary bool array
    """
    return {
        "WT": (seg > 0),            # Whole Tumor
        "TC": ((seg == 1) | (seg == 3)),  # Tumor Core
        "ET": (seg == 3),           # Enhancing Tumor
    }


# ---------------------------------------------------------------------------
# Individual metric functions (numpy, binary masks)
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice similarity coefficient for two binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Intersection-over-Union (Jaccard index) for two binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return (intersection + smooth) / (union + smooth)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sensitivity = TP / (TP + FN)"""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return tp / (tp + fn + 1e-8)


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Specificity = TN / (TN + FP)"""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum()
    return tn / (tn + fp + 1e-8)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                          voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff Distance (HD95) in mm.

    If either mask is empty, returns np.nan (handled in aggregation).
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    # Get surface voxels (voxels that have at least one background neighbour)
    def surface(mask):
        eroded = binary_erosion(mask)
        return mask & ~eroded

    pred_surf = np.argwhere(surface(pred)).astype(float)
    gt_surf   = np.argwhere(surface(gt)).astype(float)

    # Apply voxel spacing
    pred_surf = pred_surf * np.array(voxel_spacing)
    gt_surf   = gt_surf   * np.array(voxel_spacing)

    # Compute pairwise directed distances
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_surf)
    tree_pred = cKDTree(pred_surf)

    d_pred_to_gt, _ = tree_gt.query(pred_surf)
    d_gt_to_pred, _ = tree_pred.query(gt_surf)

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


# ---------------------------------------------------------------------------
# Per-patient metric computation
# ---------------------------------------------------------------------------

def compute_patient_metrics(pred_seg: np.ndarray, gt_seg: np.ndarray,
                             voxel_spacing: tuple = (1.0, 1.0, 1.0)
                             ) -> Dict[str, Dict[str, float]]:
    """
    Compute all 5 metrics for all 3 BraTS sub-regions for one patient.

    Args:
        pred_seg : integer array (H, W, D) — model prediction
        gt_seg   : integer array (H, W, D) — ground truth
        voxel_spacing : voxel size in mm (from NIfTI header)

    Returns:
        nested dict: {region: {metric: value}}
        e.g. {'WT': {'dice': 0.87, 'iou': 0.79, 'hd95': 5.1, ...}, ...}
    """
    pred_regions = get_regions(pred_seg)
    gt_regions   = get_regions(gt_seg)

    results = {}
    for region in ["WT", "TC", "ET"]:
        p = pred_regions[region]
        g = gt_regions[region]
        results[region] = {
            "dice":        dice_score(p, g),
            "iou":         iou_score(p, g),
            "hd95":        hausdorff_distance_95(p, g, voxel_spacing),
            "sensitivity": sensitivity(p, g),
            "specificity": specificity(p, g),
        }

    return results


# ---------------------------------------------------------------------------
# Aggregation across patients
# ---------------------------------------------------------------------------

def aggregate_metrics(all_results: list) -> Dict[str, Dict[str, float]]:
    """
    Average per-patient metrics across the test set.
    HD95 NaN values (empty predictions) are handled gracefully.

    Args:
        all_results : list of dicts from compute_patient_metrics()
    Returns:
        nested dict: {region: {metric: mean ± std string}} — but also returns
        raw arrays as {region: {metric: [values]}} under key '_raw'
    """
    aggregated = {}
    for region in ["WT", "TC", "ET"]:
        aggregated[region] = {}
        for metric in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
            vals = [r[region][metric] for r in all_results]
            vals_clean = [v for v in vals if not np.isnan(v)]
            if vals_clean:
                aggregated[region][metric] = {
                    "mean": float(np.mean(vals_clean)),
                    "std":  float(np.std(vals_clean)),
                    "raw":  vals_clean,
                }
            else:
                aggregated[region][metric] = {"mean": np.nan, "std": np.nan, "raw": []}

    return aggregated


def print_metrics_table(aggregated: Dict, model_name: str = "Model") -> None:
    """Pretty-print the 15-metric table (5 metrics × 3 regions)."""
    print(f"\n{'='*70}")
    print(f"  Results for: {model_name}")
    print(f"{'='*70}")
    header = f"{'Metric':<15} {'WT':>12} {'TC':>12} {'ET':>12}"
    print(header)
    print("-" * 55)

    metrics_display = [
        ("Dice ↑",        "dice"),
        ("IoU ↑",         "iou"),
        ("HD95 ↓ (mm)",   "hd95"),
        ("Sensitivity ↑", "sensitivity"),
        ("Specificity ↑", "specificity"),
    ]

    for label, key in metrics_display:
        row = f"{label:<15}"
        for region in ["WT", "TC", "ET"]:
            m = aggregated[region][key]
            row += f"  {m['mean']:.4f}±{m['std']:.3f}"
        print(row)
    print("=" * 70)
