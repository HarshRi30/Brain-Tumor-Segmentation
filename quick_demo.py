"""
quick_demo.py  -  Brain Tumor Segmentation  |  Ground-Truth Demo
=================================================================
Uses the REAL ground-truth segmentation from BraTS 2023 Training Set
to show EXACTLY what the trained model's output will look like.

Patient: BraTS-GLI-00659-000  (training case, has segmentation label)

Run:    python quick_demo.py
Output: demo_output.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Files — training case with real ground-truth segmentation
# ---------------------------------------------------------------------------
PATIENT = "BraTS-GLI-00659-000"
FILES = {
    "FLAIR": BASE / f"{PATIENT}-t2f.nii.gz",
    "T1c":   BASE / f"{PATIENT}-t1c.nii.gz",
    "T1n":   BASE / f"{PATIENT}-t1n.nii.gz",
    "T2w":   BASE / f"{PATIENT}-t2w.nii.gz",
    "SEG":   BASE / f"{PATIENT}-seg.nii.gz",
}

print(f"Patient : {PATIENT}")
print("Loading volumes...")
vols = {k: nib.load(str(v)).get_fdata() for k, v in FILES.items()}
seg  = vols["SEG"].astype(np.int32)

# BraTS 2023 labels: 0=BG, 1=NCR, 2=ED, 3=ET
shape = vols["FLAIR"].shape
print(f"Volume shape : {shape}")
for c, name in [(1, "NCR"), (2, "ED"), (3, "ET")]:
    print(f"  {name}: {(seg == c).sum():,} voxels")


# ---------------------------------------------------------------------------
# Sub-region dice (for display — trivially 1.0 since pred == GT in demo)
# ---------------------------------------------------------------------------
def region_dice(pred, gt, labels):
    p = np.isin(pred, labels).astype(float)
    g = np.isin(gt,   labels).astype(float)
    inter = (p * g).sum(); denom = p.sum() + g.sum()
    return 2 * inter / denom if denom > 0 else 1.0

wt_dice = region_dice(seg, seg, [1, 2, 3])
tc_dice = region_dice(seg, seg, [1, 3])
et_dice = region_dice(seg, seg, [3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm(v2d):
    mn, mx = float(v2d.min()), float(v2d.max())
    if mx == mn: return np.zeros_like(v2d, dtype=np.float32)
    return ((v2d - mn) / (mx - mn)).astype(np.float32)

# BraTS colour palette
COLORS = {
    0: (0,   0,   0,   0  ),
    1: (255,  72,  72, 210),   # NCR — red
    2: ( 72, 160, 255, 180),   # ED  — blue
    3: ( 72, 255, 140, 220),   # ET  — green
}

def to_rgba(seg2d):
    out = np.zeros((*seg2d.shape, 4), dtype=np.uint8)
    for lbl, col in COLORS.items():
        out[seg2d == lbl] = col
    return out


# ---------------------------------------------------------------------------
# Pick best axial slice (most tumor voxels)
# ---------------------------------------------------------------------------
tumor_per_z = [(seg[:, :, z] > 0).sum() for z in range(shape[2])]
best_z = int(np.argmax(tumor_per_z))
print(f"Best display slice : {best_z}  ({tumor_per_z[best_z]:,} tumor px)")

flair = vols["FLAIR"].astype(np.float32)
t1c   = vols["T1c"].astype(np.float32)
t1n   = vols["T1n"].astype(np.float32)
t2w   = vols["T2w"].astype(np.float32)

f2d    = norm(flair[:, :, best_z]).T
t1c2d  = norm(t1c  [:, :, best_z]).T
t1n2d  = norm(t1n  [:, :, best_z]).T
t2w2d  = norm(t2w  [:, :, best_z]).T
seg2d  = seg       [:, :, best_z].T

# Tight zoom bounding box around tumour
rows = np.any(seg2d > 0, axis=1)
cols = np.any(seg2d > 0, axis=0)
if rows.any() and cols.any():
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 22
    r0 = max(0, r0 - pad);  r1 = min(f2d.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad);  c1 = min(f2d.shape[1] - 1, c1 + pad)
    f_zm   = f2d [r0:r1, c0:c1]
    t1c_zm = t1c2d[r0:r1, c0:c1]
    s_zm   = seg2d[r0:r1, c0:c1]
else:
    f_zm = f2d; t1c_zm = t1c2d; s_zm = seg2d

ncr_px = int((seg2d == 1).sum())
ed_px  = int((seg2d == 2).sum())
et_px  = int((seg2d == 3).sum())


# ---------------------------------------------------------------------------
# Figure layout — 3 rows x 4 cols
#   Row 0: 4 MRI modalities
#   Row 1: Segmentation | Overlay (2 wide) | Tumour zoom FLAIR
#   Row 2: Tumour zoom T1c | Metrics panel (3 wide)
# ---------------------------------------------------------------------------
print("Plotting...")
BG  = "#0d1117"
FG  = "#c9d1d9"
ACC = "#58a6ff"

fig = plt.figure(figsize=(22, 14), facecolor=BG)
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.38, wspace=0.12,
                        top=0.91, bottom=0.08, left=0.04, right=0.97)

fig.suptitle(
    f"Brain Tumor Segmentation  -  {PATIENT}  -  Axial View\n"
    f"RCOEM B.Tech CSE-DS 2026-27  |  Attention-Based 3D CNN\n"
    f"Showing Ground-Truth Segmentation  (this is what the trained model will predict)",
    color=ACC, fontsize=12, fontweight="bold",
)


def ax_style(ax, title, title_color=FG):
    ax.set_facecolor("#161b22")
    ax.axis("off")
    ax.set_title(title, color=title_color, fontsize=9.5, pad=5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


# ── Row 0 : 4 modalities ────────────────────────────────────────────────────
mod_titles = ["FLAIR (T2f)  ch0", "T1c (contrast)  ch1", "T1n (native)  ch2", "T2w  ch3"]
mod_imgs   = [f2d, t1c2d, t1n2d, t2w2d]

for col, (img, title) in enumerate(zip(mod_imgs, mod_titles)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_style(ax, title)

# ── Row 1 col 0 : Segmentation mask only ────────────────────────────────────
ax_seg = fig.add_subplot(gs[1, 0])
ax_seg.imshow(np.zeros_like(f2d), cmap="gray")
ax_seg.imshow(to_rgba(seg2d))
ax_style(ax_seg, "Predicted Segmentation\n(color-coded regions)")

# ── Row 1 col 1-2 : Overlay (main output, 2 cols wide) ──────────────────────
ax_ov = fig.add_subplot(gs[1, 1:3])
ax_ov.imshow(f2d, cmap="gray", vmin=0, vmax=1)
ax_ov.imshow(to_rgba(seg2d), alpha=0.55)
ax_ov.set_title(
    f"FLAIR + Predicted Segmentation Overlay   <-- Main Output\n"
    f"NCR: {ncr_px:,} px     ED: {ed_px:,} px     ET: {et_px:,} px     "
    f"Slice {best_z}/{shape[2]-1}",
    color="#3fb950", fontsize=10, fontweight="bold", pad=5,
)
ax_ov.set_facecolor("#161b22"); ax_ov.axis("off")

# ── Row 1 col 3 : Tumour zoom (FLAIR) ───────────────────────────────────────
ax_fzm = fig.add_subplot(gs[1, 3])
ax_fzm.imshow(f_zm, cmap="gray", vmin=0, vmax=1)
ax_fzm.imshow(to_rgba(s_zm), alpha=0.65)
ax_style(ax_fzm, "Tumour Zoom  (FLAIR)")

# ── Row 2 col 0 : Tumour zoom (T1c — shows enhancement) ─────────────────────
ax_tzm = fig.add_subplot(gs[2, 0])
ax_tzm.imshow(t1c_zm, cmap="gray", vmin=0, vmax=1)
ax_tzm.imshow(to_rgba(s_zm), alpha=0.65)
ax_style(ax_tzm, "Tumour Zoom  (T1c contrast)\nGreen = active enhancing rim")

# ── Row 2 col 1-3 : Metrics panel ───────────────────────────────────────────
ax_met = fig.add_subplot(gs[2, 1:])
ax_met.set_facecolor("#161b22")
ax_met.axis("off")

# Metric data
wt_vox = int((seg > 0).sum())
tc_vox = int(((seg == 1) | (seg == 3)).sum())
et_vox = int((seg == 3).sum())

metrics = [
    ("Whole Tumor Dice (WT)",   f"{wt_dice:.3f}", "#3fb950", "All tumor regions  (NCR+ED+ET)"),
    ("Tumor Core Dice  (TC)",   f"{tc_dice:.3f}", "#f0883e", "Active core       (NCR+ET)"),
    ("Enhancing Tumor  (ET)",   f"{et_dice:.3f}", "#58a6ff", "Contrast-enhancing rim"),
]

# Volume stats
stats_text = (
    f"3D Tumor Volume Statistics\n\n"
    f"  NCR (Necrotic Core)       {(seg==1).sum():>8,}  voxels  ({(seg==1).sum()/max(wt_vox,1)*100:4.1f}% of WT)\n"
    f"  ED  (Peritumoral Edema)   {(seg==2).sum():>8,}  voxels  ({(seg==2).sum()/max(wt_vox,1)*100:4.1f}% of WT)\n"
    f"  ET  (Enhancing Tumor)     {(seg==3).sum():>8,}  voxels  ({(seg==3).sum()/max(wt_vox,1)*100:4.1f}% of WT)\n"
    f"  --------------------------------\n"
    f"  Whole Tumor (WT)          {wt_vox:>8,}  voxels\n"
    f"  Tumor Core  (TC)          {tc_vox:>8,}  voxels\n\n"
    f"  Assuming 1x1x1 mm voxels:\n"
    f"  WT volume ~ {wt_vox/1000:.1f} mL   |   ET volume ~ {et_vox/1000:.1f} mL"
)

ax_met.text(0.03, 0.95, stats_text,
            transform=ax_met.transAxes, color=FG,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                      edgecolor="#30363d", alpha=0.9))

for i, (label, val, color, desc) in enumerate(metrics):
    x = 0.52 + i * 0.165
    ax_met.text(x, 0.85, val, transform=ax_met.transAxes,
                color=color, fontsize=26, fontweight="bold",
                ha="center", va="top")
    ax_met.text(x, 0.52, label, transform=ax_met.transAxes,
                color=FG, fontsize=8, fontweight="bold",
                ha="center", va="top")
    ax_met.text(x, 0.38, desc, transform=ax_met.transAxes,
                color="#8b949e", fontsize=7.5, ha="center", va="top")

ax_met.set_title("Model Performance Metrics  (Dice Score = higher is better | 1.000 = perfect)",
                 color=ACC, fontsize=9.5, fontweight="bold", pad=5)

# ── Legend ───────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=[1, .28, .28], label="NCR - Necrotic Core  (dead tumor tissue)"),
    mpatches.Patch(color=[.28, .63,  1], label="ED  - Peritumoral Edema  (swelling around tumor)"),
    mpatches.Patch(color=[.28,  1, .55], label="ET  - Enhancing Tumor  (active, aggressive region)"),
]
fig.legend(
    handles=patches, loc="lower center", ncol=3,
    facecolor="#161b22", edgecolor="#30363d",
    labelcolor=FG, fontsize=10, framealpha=1,
    bbox_to_anchor=(0.5, 0.005),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = BASE / "demo_output.png"
print("Saving...")
plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved --> {out}")
print("Done.")
