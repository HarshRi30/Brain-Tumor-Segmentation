"""
losses.py — Loss functions for brain tumor segmentation.

Implements:
  - DiceLoss      : per-class Dice averaged across foreground classes
  - FocalLoss     : handles class imbalance better than plain CE
  - CombinedLoss  : weighted sum of Dice + Focal (default loss used in training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss averaged over foreground classes.
    Works with multi-class segmentation (one-hot encoded internally).

    Args:
        num_classes  : total number of classes (including background)
        smooth       : Laplace smoothing to avoid division by zero
        ignore_bg    : if True, background (class 0) is excluded from average
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5,
                 ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D) — raw model output (before softmax)
            targets : (B, H, W, D)    — integer class labels
        Returns:
            scalar Dice loss
        """
        probs = F.softmax(logits, dim=1)                       # (B, C, H, W, D)

        # One-hot encode targets → (B, C, H, W, D)
        targets_oh = F.one_hot(targets, self.num_classes)      # (B, H, W, D, C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float() # (B, C, H, W, D)

        # Dice per class
        dims = (0, 2, 3, 4)  # sum over batch + spatial dims
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality   = (probs + targets_oh).sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over foreground classes only
        start = 1 if self.ignore_bg else 0
        loss = 1.0 - dice_per_class[start:].mean()
        return loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard / rare classes.
    Particularly helpful for the small Enhancing Tumor region.

    Args:
        gamma   : focusing parameter (2.0 is standard)
        alpha   : class weights tensor of shape (C,), or None for uniform
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (C,) tensor or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D)
            targets : (B, H, W, D)
        """
        B, C, H, W, D = logits.shape
        logits_flat  = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)
        targets_flat = targets.reshape(-1)                             # (N,)

        log_probs = F.log_softmax(logits_flat, dim=1)           # (N, C)
        probs     = torch.exp(log_probs)                        # (N, C)

        # Gather log-prob for the true class
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # (N,)
        pt     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)      # (N,)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets_flat]
            focal_weight = focal_weight * alpha_t

        loss = -(focal_weight * log_pt).mean()
        return loss


# ---------------------------------------------------------------------------
# Combined Loss (default)
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted sum of DiceLoss + FocalLoss.
    This combination consistently outperforms either loss alone for BraTS.

    Args:
        num_classes : total classes (4 for BraTS 2023)
        dice_weight : weight for Dice component
        focal_weight: weight for Focal component
        gamma       : Focal loss focusing parameter
    """

    def __init__(self, num_classes: int = 4,
                 dice_weight: float = 0.5, focal_weight: float = 0.5,
                 gamma: float = 2.0):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss  = DiceLoss(num_classes=num_classes)
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, dice_loss, focal_loss  (all scalar tensors)
        """
        dl = self.dice_loss(logits, targets)
        fl = self.focal_loss(logits, targets)
        total = self.dice_weight * dl + self.focal_weight * fl
        return total, dl, fl
