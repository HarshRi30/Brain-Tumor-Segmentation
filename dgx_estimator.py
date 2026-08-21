#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DGX Resource & Work-Hour Estimator — Brain Tumor Segmentation       ║
║          RCOEM B.Tech CSE-DS 2026-27 | BraTS 2023 GLI                       ║
║                                                                              ║
║  Run this script on the DGX BEFORE training to get accurate estimates of:   ║
║    • GPU VRAM usage per training configuration                               ║
║    • Training time per notebook / phase                                      ║
║    • Storage requirements (raw + preprocessed + checkpoints)                ║
║    • Recommended DGX session schedule                                        ║
║    • CPU / RAM bottleneck warnings                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python dgx_estimator.py                        # uses paths from src/config.py
    python dgx_estimator.py --dataset /path/to/brats --gpu-index 0
    python dgx_estimator.py --full-scan            # count actual dataset files
    python dgx_estimator.py --export report.txt    # save report to file
    python dgx_estimator.py --json report.json     # machine-readable output

Requirements (already in requirements.txt):
    torch, psutil, numpy
"""

import argparse
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── optional imports (graceful fallback) ─────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ── Force UTF-8 on Windows so box-drawing characters render correctly ────────
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── ANSI colours (disabled automatically if stdout is not a TTY) ──────────────
USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD    = lambda t: _c("1",  t)
GREEN   = lambda t: _c("92", t)
YELLOW  = lambda t: _c("93", t)
RED     = lambda t: _c("91", t)
CYAN    = lambda t: _c("96", t)
MAGENTA = lambda t: _c("95", t)
DIM     = lambda t: _c("2",  t)
BLUE    = lambda t: _c("94", t)


# =============================================================================
# 1.  HARDWARE PROBE
# =============================================================================

class HardwareInfo:
    """Collect live hardware information from the DGX node."""

    def __init__(self, gpu_index=0):
        self.gpu_index    = gpu_index
        self.gpus         = []
        self.cpu_cores    = 0
        self.cpu_model    = "Unknown"
        self.ram_total_gb = 0.0
        self.ram_avail_gb = 0.0
        self.disk_info    = {}
        self._probe()

    # ── GPU ──────────────────────────────────────────────────────────────────
    def _probe_gpus_torch(self):
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            self.gpus.append({
                "index":         i,
                "name":          props.name,
                "vram_total_gb": props.total_memory / 1e9,
                "vram_free_gb":  None,
                "sm_count":      props.multi_processor_count,
                "compute_cap":   f"{props.major}.{props.minor}",
                "temp_c":        "—",
                "util_pct":      "—",
            })
        return True

    def _probe_gpus_nvidiasmi(self):
        """nvidia-smi fallback / supplement for free VRAM."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode()
        except Exception:
            return
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            idx      = int(parts[0])
            total_gb = float(parts[2]) / 1024
            free_gb  = float(parts[3]) / 1024
            temp     = parts[4]
            util     = parts[5]
            matched  = False
            for g in self.gpus:
                if g["index"] == idx:
                    g["vram_free_gb"] = free_gb
                    g["temp_c"]       = temp
                    g["util_pct"]     = util
                    matched = True
                    break
            if not matched:
                self.gpus.append({
                    "index":         idx,
                    "name":          parts[1],
                    "vram_total_gb": total_gb,
                    "vram_free_gb":  free_gb,
                    "temp_c":        temp,
                    "util_pct":      util,
                    "sm_count":      None,
                    "compute_cap":   None,
                })

    # ── CPU / RAM ─────────────────────────────────────────────────────────────
    def _probe_cpu_ram(self):
        if PSUTIL_AVAILABLE:
            self.cpu_cores    = psutil.cpu_count(logical=True)
            mem               = psutil.virtual_memory()
            self.ram_total_gb = mem.total    / 1e9
            self.ram_avail_gb = mem.available / 1e9
        else:
            try:
                with open("/proc/cpuinfo") as f:
                    self.cpu_cores = f.read().count("processor\t:")
            except Exception:
                self.cpu_cores = os.cpu_count() or 1
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            self.ram_total_gb = int(line.split()[1]) / 1e6
                        if line.startswith("MemAvailable"):
                            self.ram_avail_gb = int(line.split()[1]) / 1e6
            except Exception:
                pass

        try:
            if platform.system() == "Linux":
                out = subprocess.check_output(
                    "grep 'model name' /proc/cpuinfo | head -1",
                    shell=True, stderr=subprocess.DEVNULL, timeout=5,
                ).decode()
                self.cpu_model = out.split(":")[1].strip() if ":" in out else "Unknown"
            elif platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                )
                self.cpu_model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except Exception:
            pass

    # ── Disk ──────────────────────────────────────────────────────────────────
    def _probe_disk(self):
        if PSUTIL_AVAILABLE:
            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    self.disk_info[part.mountpoint] = {
                        "total_gb": usage.total / 1e9,
                        "free_gb":  usage.free  / 1e9,
                    }
                except Exception:
                    pass

    def _probe(self):
        self._probe_gpus_torch()
        self._probe_gpus_nvidiasmi()
        self._probe_cpu_ram()
        self._probe_disk()

    @property
    def primary_gpu(self):
        for g in self.gpus:
            if g["index"] == self.gpu_index:
                return g
        return self.gpus[0] if self.gpus else {}

    @property
    def num_gpus(self):
        return len(self.gpus)


# =============================================================================
# 2.  DATASET SCANNER
# =============================================================================

class DatasetInfo:
    """Scan the BraTS 2023 GLI dataset folder (or estimate from size)."""

    FILES_PER_PATIENT = 5   # 4 modalities + 1 seg

    def __init__(self, dataset_path, full_scan=False):
        self.path       = Path(dataset_path)
        self.full_scan  = full_scan
        self.exists     = self.path.exists()
        self.n_patients = 0
        self.raw_size_gb = 0.0
        self.scan()

    def scan(self):
        if not self.exists:
            return
        try:
            subdirs = [d for d in self.path.iterdir() if d.is_dir()]
            self.n_patients = len(subdirs)
        except Exception:
            return

        if self.full_scan:
            total_bytes = sum(
                f.stat().st_size
                for f in self.path.rglob("*.nii.gz")
                if f.is_file()
            )
            self.raw_size_gb = total_bytes / 1e9
        else:
            # ~30 MB per patient (estimate for BraTS 2023 compressed .nii.gz)
            self.raw_size_gb = self.n_patients * 30 / 1024


# =============================================================================
# 3.  VRAM ESTIMATOR
# =============================================================================

class VRAMEstimator:
    """
    Estimate GPU VRAM consumption for 3D U-Net variants on BraTS patches.

    Approach:
      total VRAM = model_params + optimizer_states + activations + input + overhead
    """

    DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

    # Approximate parameter counts in millions (from architecture analysis)
    MODEL_PARAMS_M = {
        "3D U-Net (Baseline)":       8.6,
        "Attention U-Net 3D (Ours)": 11.2,
        "Channel Attention only":    9.4,
        "Spatial Attention only":    10.1,
    }

    def __init__(self, patch_size=(96, 96, 96), batch_size=2, init_features=32,
                 in_channels=4, out_channels=4, dtype="fp32"):
        self.patch  = patch_size
        self.bs     = batch_size
        self.feats  = init_features
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.dtype  = dtype
        self._db    = self.DTYPE_BYTES[dtype]

    def _activation_vram_gb(self, model_name):
        p = self.patch
        voxels = p[0] * p[1] * p[2]
        F = self.feats
        # Encoder: 4 levels, features double each level
        enc = (
            voxels         * F     +
            voxels // 8    * F * 2 +
            voxels // 64   * F * 4 +
            voxels // 512  * F * 8
        )
        dec = enc  # decoder mirrors encoder
        total = (enc + dec) * 2  # forward + backward gradients
        attn_factor = 1.20 if "Attention" in model_name else 1.0
        return (total * self._db * self.bs * attn_factor) / 1e9

    def _param_vram_gb(self, model_name):
        params_m = self.MODEL_PARAMS_M[model_name]
        # params + Adam (m, v buffers) + gradients = 4× params in fp32
        return (params_m * 1e6 * self._db * 4) / 1e9

    def estimate(self, model_name):
        param_gb   = self._param_vram_gb(model_name)
        act_gb     = self._activation_vram_gb(model_name)
        inp_gb     = (math.prod(self.patch) * self.in_ch * self.bs * self._db) / 1e9
        overhead_gb = (param_gb + act_gb + inp_gb) * 0.10
        total_gb   = param_gb + act_gb + inp_gb + overhead_gb
        return {
            "model":          model_name,
            "params_gb":      round(param_gb,   2),
            "activations_gb": round(act_gb,     2),
            "input_gb":       round(inp_gb,     3),
            "overhead_gb":    round(overhead_gb, 2),
            "total_gb":       round(total_gb,   2),
        }

    def estimate_all(self):
        return [self.estimate(m) for m in self.MODEL_PARAMS_M]


# =============================================================================
# 4.  TRAINING TIME ESTIMATOR
# =============================================================================

class TimeEstimator:
    """
    Estimate wall-clock training time based on GPU tier and project config.

    Benchmarks (iterations/sec for 96³ patch, bs=2, fp32) are derived from
    published MONAI 3D segmentation benchmarks on BraTS-class datasets.
    """

    GPU_ITERS_PER_SEC = {
        "A100":     5.8,
        "H100":     9.5,
        "A40":      4.1,
        "V100":     3.2,
        "RTX 4090": 4.5,
        "RTX 3090": 2.8,
        "Unknown":  3.0,
    }

    PREPROCESS_SEC_PER_PATIENT = {
        "A100":     3.5,
        "H100":     2.8,
        "A40":      4.2,
        "V100":     5.0,
        "RTX 4090": 4.0,
        "RTX 3090": 6.0,
        "Unknown":  5.0,
    }

    def __init__(self, hw, n_patients, patch_size=(96, 96, 96),
                 batch_size=2, num_epochs=150, n_folds=3,
                 patches_per_volume=2, train_ratio=0.8):
        self.hw          = hw
        self.n_patients  = n_patients
        self.patch       = patch_size
        self.bs          = batch_size
        self.epochs      = num_epochs
        self.n_folds     = n_folds
        self.ppv         = patches_per_volume
        self.train_ratio = train_ratio
        self.gpu_tier    = self._detect_gpu_tier()

    def _detect_gpu_tier(self):
        gpu  = self.hw.primary_gpu
        name = gpu.get("name", "Unknown").upper()
        for tier in self.GPU_ITERS_PER_SEC:
            if tier.upper() in name:
                return tier
        # Heuristic by VRAM size
        vram = gpu.get("vram_total_gb", 0)
        if vram >= 75: return "A100"
        if vram >= 40: return "A40"
        if vram >= 32: return "V100"
        if vram >= 24: return "RTX 4090"
        if vram >= 16: return "RTX 3090"
        return "Unknown"

    @property
    def iters_per_sec(self):
        return self.GPU_ITERS_PER_SEC[self.gpu_tier]

    def _train_time_hours(self, model_name, n_patients=None):
        n       = n_patients or self.n_patients
        n_train = int(n * self.train_ratio)
        iters_per_epoch = (n_train * self.ppv) / self.bs
        total_iters     = iters_per_epoch * self.epochs * self.n_folds
        overhead = 1.25 if "Attention" in model_name else 1.0
        return (total_iters * overhead) / (self.iters_per_sec * 3600)

    def _preprocess_time_hours(self, n_patients=None):
        n   = n_patients or self.n_patients
        sec = self.PREPROCESS_SEC_PER_PATIENT[self.gpu_tier]
        return (n * sec) / 3600

    def estimate_all(self):
        models = {
            "3D U-Net (Baseline)":       self._train_time_hours("3D U-Net"),
            "Attention U-Net 3D (Ours)": self._train_time_hours("Attention U-Net"),
            "Channel Attention only":    self._train_time_hours("Attention Channel") * 0.85,
            "Spatial Attention only":    self._train_time_hours("Attention Spatial") * 0.90,
        }
        preprocess  = self._preprocess_time_hours()
        evaluation  = 0.5
        ablation    = models["Channel Attention only"] + models["Spatial Attention only"]
        visualise   = 0.5
        total_gpu   = preprocess + sum(models.values()) + evaluation + visualise

        human_hours = {
            "Environment setup & WinSCP transfer":      2.0,
            "Dataset inspection & EDA notebook":        1.5,
            "Pre-processing notebook (01)":             0.5,
            "Baseline training monitoring (03)":        1.0,
            "Attention U-Net training monitoring (04)": 1.0,
            "Evaluation & ablation notebooks (05,06)":  2.0,
            "Visualisation & Grad-CAM (07)":            1.5,
            "Report writing & result tables":           6.0,
            "Buffer / debugging":                       2.0,
        }

        return {
            "gpu_tier":        self.gpu_tier,
            "iters_per_sec":   self.iters_per_sec,
            "preprocess_hrs":  round(preprocess, 2),
            "models":          {k: round(v, 2) for k, v in models.items()},
            "evaluation_hrs":  evaluation,
            "ablation_hrs":    round(ablation, 2),
            "visualise_hrs":   visualise,
            "total_gpu_hrs":   round(total_gpu, 2),
            "human_hours":     human_hours,
            "total_human_hrs": round(sum(human_hours.values()), 1),
        }


# =============================================================================
# 5.  STORAGE ESTIMATOR
# =============================================================================

class StorageEstimator:
    """Estimate disk space requirements for each project stage."""

    def __init__(self, n_patients, patch_size=(96, 96, 96),
                 patches_per_volume=2, num_epochs=150, n_folds=3):
        self.n     = n_patients
        self.p     = patch_size
        self.ppv   = patches_per_volume
        self.ep    = num_epochs
        self.folds = n_folds

    def estimate(self):
        # Raw BraTS: ~30 MB/patient compressed
        raw_gb = self.n * 30 / 1024

        # Preprocessed patches: 96³ × 4ch × fp32 × 2 patches/vol
        patch_bytes    = math.prod(self.p) * 4 * 4 * self.ppv
        preprocessed_gb = self.n * patch_bytes / 1e9

        # Checkpoints: ~45 MB each; save every 5 epochs × folds × 2 models
        saves          = (self.ep // 5) * self.folds * 2
        checkpoints_gb = saves * 45 / 1024

        logs_gb    = 0.5
        results_gb = 0.2
        total_gb   = raw_gb + preprocessed_gb + checkpoints_gb + logs_gb + results_gb

        return {
            "raw_dataset_gb":  round(raw_gb,          1),
            "preprocessed_gb": round(preprocessed_gb, 1),
            "checkpoints_gb":  round(checkpoints_gb,  1),
            "logs_gb":         round(logs_gb,          2),
            "results_gb":      round(results_gb,       2),
            "total_gb":        round(total_gb,          1),
        }


# =============================================================================
# 6.  DGX SESSION PLANNER
# =============================================================================

def build_session_plan(time_est, slot_hours=4.0):
    return [
        ("Day 1", "01_preprocess.ipynb",
         time_est["preprocess_hrs"],
         "Run once; outputs saved permanently"),
        ("Day 1", "02_explore_data.ipynb",
         0.33,
         "EDA, class balance, sample slices"),
        ("Day 2", "03_train_baseline.ipynb",
         time_est["models"]["3D U-Net (Baseline)"],
         "Baseline — checkpoints every 5 epochs"),
        ("Day 3", "Resume baseline if needed",
         0.0,
         "Auto-resumes from last checkpoint"),
        ("Day 3", "04_train_attention_unet.ipynb",
         time_est["models"]["Attention U-Net 3D (Ours)"],
         "Proposed model — resumes next session"),
        ("Day 4", "Resume Attention U-Net",
         0.0,
         "Checkpoint auto-resume"),
        ("Day 5", "05_evaluate_compare.ipynb",
         0.5,
         "Metrics table: Dice / IoU / HD95"),
        ("Day 5", "06_ablation_study.ipynb",
         time_est["ablation_hrs"],
         "Channel vs Spatial vs Hybrid"),
        ("Day 6", "07_visualise_results.ipynb",
         0.5,
         "Attention maps, Grad-CAM, overlays"),
        ("Day 6", "Demo & cleanup",
         0.25,
         "Package results for report"),
    ]


# =============================================================================
# 7.  REPORT PRINTER
# =============================================================================

def fmt_gb(v, warn_above=0, crit_above=0):
    s = f"{v:.1f} GB"
    if crit_above and v >= crit_above:
        return RED(s)
    if warn_above and v >= warn_above:
        return YELLOW(s)
    return GREEN(s)


def fmt_hrs(h):
    hours = int(h)
    mins  = int((h - hours) * 60)
    if hours == 0:
        return YELLOW(f"{mins}m")
    if mins == 0:
        return YELLOW(f"{hours}h")
    return YELLOW(f"{hours}h {mins}m")


def status_bar(used, total, width=24):
    if total <= 0:
        return "[" + "?" * width + "]"
    pct    = min(used / total, 1.0)
    filled = int(pct * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = GREEN if pct < 0.70 else (YELLOW if pct < 0.90 else RED)
    return f"[{color(bar)}] {pct * 100:.0f}%"


def strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*m', '', text)


def print_report(hw, ds, vram_est, time_res, storage, args, output_lines):
    def emit(*parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        output_lines.append(strip_ansi(line))

    # ── Header ────────────────────────────────────────────────────────────────
    emit()
    emit(BOLD("╔" + "═" * 78 + "╗"))
    emit(BOLD("║") + MAGENTA("  DGX Resource & Work-Hour Estimator").center(78) + BOLD("║"))
    emit(BOLD("║") + "  Brain Tumor Segmentation · RCOEM B.Tech CSE-DS 2026-27".center(78) + BOLD("║"))
    emit(BOLD("║") + f"  Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}".ljust(78) + BOLD("║"))
    emit(BOLD("╚" + "═" * 78 + "╝"))

    # ── 1. Hardware ───────────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("1 · HARDWARE DETECTED")))
    emit(CYAN("─" * 80))
    emit(f"  {'CPU':<28} {hw.cpu_model[:47]}")
    emit(f"  {'CPU cores (logical)':<28} {hw.cpu_cores}")
    emit(f"  {'System RAM':<28} {hw.ram_total_gb:.0f} GB total  |  {hw.ram_avail_gb:.1f} GB available")
    emit()
    if hw.gpus:
        emit(f"  {'#':<4}{'GPU Name':<30}{'VRAM Total':>12}{'VRAM Free':>12}{'Temp':>8}{'Util':>8}")
        emit("  " + "─" * 76)
        for g in hw.gpus:
            free_str  = f"{g['vram_free_gb']:.1f} GB" if g.get("vram_free_gb") is not None else "—"
            temp_str  = f"{g.get('temp_c', '?')} °C"
            util_str  = f"{g.get('util_pct', '?')} %"
            vt        = g.get("vram_total_gb", 0)
            star      = MAGENTA("  ◄ PRIMARY") if g["index"] == args.gpu_index else ""
            emit(f"  {g['index']:<4}{g['name']:<30}{vt:>9.1f} GB{free_str:>12}{temp_str:>8}{util_str:>8}{star}")
    else:
        emit(RED("  ⚠  No NVIDIA GPU detected — install CUDA drivers and retry"))

    if hw.disk_info:
        emit()
        emit(f"  {'Mount':<20}{'Total':>10}{'Free':>10}")
        emit("  " + "─" * 42)
        for mount, d in list(hw.disk_info.items())[:6]:
            emit(f"  {mount:<20}{d['total_gb']:>8.0f} GB{d['free_gb']:>8.0f} GB")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("2 · DATASET")))
    emit(CYAN("─" * 80))
    n_eff = ds.n_patients if ds.n_patients > 0 else args.n_patients
    if ds.exists:
        emit(f"  {'Path':<16} {ds.path}")
        emit(f"  {'Patients found':<16} {GREEN(str(ds.n_patients))}")
        size_note = " (exact)" if args.full_scan else "  (estimated — use --full-scan for exact)"
        emit(f"  {'Raw dataset size':<16} ~{ds.raw_size_gb:.1f} GB{size_note}")
    else:
        emit(RED(f"  ⚠  Dataset path not found: {ds.path}"))
        emit(f"  Using estimate of {GREEN(str(n_eff))} patients from config / --n-patients flag")
    emit(f"  {'Config cap':<16} {n_eff} patients (MAX_PATIENTS in src/config.py)")

    # ── 3. VRAM ───────────────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("3 · GPU VRAM ESTIMATES  [patch=96³, batch=2, fp32]")))
    emit(CYAN("─" * 80))
    gpu_vram    = hw.primary_gpu.get("vram_total_gb", 0)
    vram_table  = vram_est.estimate_all()
    emit(f"  {'Model':<32}{'Params':>8}{'Activ.':>8}{'Input':>8}{'OH':>6}{'TOTAL':>9}  Capacity")
    emit("  " + "─" * 80)
    for est in vram_table:
        total = est["total_gb"]
        if gpu_vram > 0:
            if total < gpu_vram * 0.70:
                fits = GREEN("✓ OK")
            elif total < gpu_vram * 0.90:
                fits = YELLOW("⚠ TIGHT")
            elif total < gpu_vram:
                fits = YELLOW("⚠ VERY TIGHT")
            else:
                fits = RED("✗ OOM RISK")
            bar = status_bar(total, gpu_vram, width=16)
        else:
            fits = DIM("GPU unknown")
            bar  = ""
        emit(
            f"  {est['model']:<32}{est['params_gb']:>6.1f} GB{est['activations_gb']:>6.1f} GB"
            f"{est['input_gb']:>6.2f} GB{est['overhead_gb']:>4.1f} GB{total:>7.1f} GB  {fits}"
        )
    emit()
    if gpu_vram > 0:
        emit(f"  GPU VRAM available: {fmt_gb(gpu_vram)} on {hw.primary_gpu.get('name', '?')}")
    emit(f"  {DIM('Tip: reduce BATCH_SIZE → 1 or PATCH_SIZE → (80,80,80) if you hit OOM')}")

    # ── 4. Training Time ──────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN(
        f"4 · TRAINING TIME  [GPU tier: {time_res['gpu_tier']} | {time_res['iters_per_sec']} iter/s]"
    )))
    emit(CYAN("─" * 80))
    emit(f"  Pre-processing (01_preprocess.ipynb)          {fmt_hrs(time_res['preprocess_hrs'])}")
    emit()
    emit(f"  {'Notebook / Model':<44}{'GPU Hours':>12}")
    emit("  " + "─" * 58)
    for model, hrs in time_res["models"].items():
        wc = f"{hrs:.1f}h ({int(hrs)}h {int((hrs % 1) * 60)}m)"
        emit(f"  {model:<44}{fmt_hrs(hrs):>12}   {DIM(wc)}")
    emit()
    emit(f"  {'Evaluation + metrics (05)':<44}{fmt_hrs(time_res['evaluation_hrs']):>12}")
    emit(f"  {'Ablation study (06)':<44}{fmt_hrs(time_res['ablation_hrs']):>12}")
    emit(f"  {'Visualisation + Grad-CAM (07)':<44}{fmt_hrs(time_res['visualise_hrs']):>12}")
    emit("  " + "─" * 58)
    emit(f"  {'TOTAL GPU-COMPUTE HOURS':<44}{BOLD(fmt_hrs(time_res['total_gpu_hrs'])):>12}")

    # ── 5. Human Work Hours ───────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("5 · HUMAN WORK HOURS")))
    emit(CYAN("─" * 80))
    emit(f"  {'Task':<48}{'Hours':>8}")
    emit("  " + "─" * 58)
    for task, hrs in time_res["human_hours"].items():
        emit(f"  {task:<48}{fmt_hrs(hrs):>8}")
    emit("  " + "─" * 58)
    emit(f"  {'TOTAL HUMAN HOURS':<48}{BOLD(fmt_hrs(time_res['total_human_hrs'])):>8}")
    emit()
    emit(DIM("  (Human hours = setup, monitoring, analysis, report writing; not GPU idle time)"))

    # ── 6. Storage ────────────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("6 · STORAGE REQUIREMENTS")))
    emit(CYAN("─" * 80))
    items = [
        ("Raw BraTS dataset",      storage["raw_dataset_gb"]),
        ("Preprocessed patches",   storage["preprocessed_gb"]),
        ("Model checkpoints",      storage["checkpoints_gb"]),
        ("TensorBoard / W&B logs", storage["logs_gb"]),
        ("Results & figures",      storage["results_gb"]),
    ]
    emit(f"  {'Stage':<36}{'Size':>10}")
    emit("  " + "─" * 48)
    for label, gb in items:
        emit(f"  {label:<36}{fmt_gb(gb):>10}")
    emit("  " + "─" * 48)
    best_free   = max((d["free_gb"] for d in hw.disk_info.values()), default=0)
    total_needed = storage["total_gb"]
    emit(f"  {'TOTAL NEEDED':<36}{BOLD(fmt_gb(total_needed)):>10}")
    emit(f"  {'DGX free disk space':<36}{fmt_gb(best_free, warn_above=total_needed * 1.5, crit_above=total_needed):>10}")
    if best_free > 0:
        if best_free < total_needed:
            emit(RED("  ⚠  INSUFFICIENT DISK SPACE — free up space before running"))
        elif best_free < total_needed * 2:
            emit(YELLOW("  ⚠  Disk space is tight — monitor closely during training"))
        else:
            emit(GREEN("  ✓  Sufficient disk space available"))

    # ── 7. Session Plan ───────────────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("7 · RECOMMENDED DGX SESSION PLAN  (6 lab slots)")))
    emit(CYAN("─" * 80))
    tasks       = build_session_plan(time_res, slot_hours=args.slot_hours)
    current_day = None
    for day, notebook, hrs, note in tasks:
        if day != current_day:
            emit()
            emit(f"  {BOLD(MAGENTA(day))}")
            current_day = day
        time_str = fmt_hrs(hrs) if hrs > 0 else DIM("(auto-resumed)")
        emit(f"    {CYAN('▸')} {notebook:<48}{time_str}  {DIM(note)}")

    # ── 8. Quick-Start Checklist ──────────────────────────────────────────────
    emit()
    emit(CYAN("─" * 80))
    emit(CYAN("  ") + BOLD(CYAN("8 · QUICK-START CHECKLIST")))
    emit(CYAN("─" * 80))
    checklist = [
        ("Transfer project folder via WinSCP",               True),
        ("Transfer BraTS dataset (.nii.gz) via WinSCP",      ds.exists),
        ("Update DATASET_PATH in src/config.py",             False),
        ("pip install -r requirements.txt",                   False),
        ("Run 01_preprocess.ipynb — Quick Test first!",       False),
        ("Verify checkpoint saves after epoch 1",             False),
        ("Monitor GPU: watch -n5 nvidia-smi",                 False),
        ("Start TensorBoard: tensorboard --logdir ./logs",    False),
    ]
    for item, done in checklist:
        icon = GREEN("☑") if done else YELLOW("☐")
        emit(f"  {icon}  {item}")

    # ── 9. Warnings ───────────────────────────────────────────────────────────
    warnings = []
    gpu = hw.primary_gpu
    if gpu:
        vram_total = gpu.get("vram_total_gb", 0)
        if 0 < vram_total < 24:
            warnings.append(
                f"Low VRAM ({vram_total:.0f} GB) — set PATCH_SIZE=(80,80,80) and BATCH_SIZE=1 in config.py"
            )
    if hw.ram_total_gb > 0 and hw.ram_total_gb < 64:
        warnings.append(
            f"System RAM ({hw.ram_total_gb:.0f} GB) < 64 GB — reduce NUM_WORKERS to 2 in config.py"
        )
    if not hw.gpus:
        warnings.append("No GPU detected — CUDA drivers must be installed to train 3D models")

    if warnings:
        emit()
        emit(CYAN("─" * 80))
        emit(CYAN("  ") + BOLD(CYAN("9 · WARNINGS")))
        emit(CYAN("─" * 80))
        for w in warnings:
            emit(RED(f"  ⚠  {w}"))

    # ── Footer ────────────────────────────────────────────────────────────────
    emit()
    emit(CYAN("═" * 80))
    emit(DIM("  Report generated by dgx_estimator.py  |  Antigravity AI · RCOEM 2026-27"))
    emit(CYAN("═" * 80))
    emit()


# =============================================================================
# 8.  ARGUMENT PARSING
# =============================================================================

def parse_args():
    # Try to load defaults from src/config.py
    default_dataset    = "/home/yourname/BraTS2023_Training_Data"
    default_n_patients = 600
    default_patch      = [96, 96, 96]
    default_batch      = 2
    default_epochs     = 150
    default_folds      = 3

    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        import config as cfg
        default_dataset    = getattr(cfg, "DATASET_PATH",       default_dataset)
        default_n_patients = getattr(cfg, "MAX_PATIENTS",       default_n_patients)
        default_patch      = list(getattr(cfg, "PATCH_SIZE",    default_patch))
        default_batch      = getattr(cfg, "BATCH_SIZE",         default_batch)
        default_epochs     = getattr(cfg, "NUM_EPOCHS",         default_epochs)
        default_folds      = getattr(cfg, "N_FOLDS",            default_folds)
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="DGX Resource & Work-Hour Estimator — Brain Tumor Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset",    default=default_dataset,
                   help="Path to BraTS training data root on DGX")
    p.add_argument("--n-patients", type=int, default=default_n_patients,
                   help="Patient count override (used when dataset path is not found)")
    p.add_argument("--gpu-index",  type=int, default=0,
                   help="Primary GPU index to plan for (default: 0)")
    p.add_argument("--patch-size", type=int, nargs=3, default=default_patch,
                   metavar=("D", "H", "W"),
                   help="Training patch size D H W (default: 96 96 96)")
    p.add_argument("--batch-size", type=int, default=default_batch,
                   help="Training batch size (default: 2)")
    p.add_argument("--epochs",     type=int, default=default_epochs,
                   help="Number of training epochs (default: 150)")
    p.add_argument("--n-folds",    type=int, default=default_folds,
                   help="Number of cross-validation folds (default: 3)")
    p.add_argument("--slot-hours", type=float, default=4.0,
                   help="Length of one DGX lab slot in hours (default: 4)")
    p.add_argument("--full-scan",  action="store_true",
                   help="Walk all .nii.gz files for exact dataset size (slow)")
    p.add_argument("--export",     default=None, metavar="FILE",
                   help="Save plain-text report to FILE")
    p.add_argument("--json",       default=None, metavar="FILE",
                   help="Export all estimates to a JSON file")
    return p.parse_args()


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    args = parse_args()

    print(f"\n{CYAN('Probing hardware...')}", flush=True)
    hw = HardwareInfo(gpu_index=args.gpu_index)

    print(f"{CYAN('Scanning dataset...')}", flush=True)
    ds = DatasetInfo(dataset_path=args.dataset, full_scan=args.full_scan)

    n_eff  = ds.n_patients if ds.n_patients > 0 else args.n_patients
    patch  = tuple(args.patch_size)

    vram_est = VRAMEstimator(
        patch_size=patch,
        batch_size=args.batch_size,
        in_channels=4, out_channels=4, init_features=32,
    )

    time_obj = TimeEstimator(
        hw=hw,
        n_patients=n_eff,
        patch_size=patch,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        n_folds=args.n_folds,
        patches_per_volume=2,
    )
    time_res = time_obj.estimate_all()

    storage_obj = StorageEstimator(
        n_patients=n_eff,
        patch_size=patch,
        patches_per_volume=2,
        num_epochs=args.epochs,
        n_folds=args.n_folds,
    )
    storage = storage_obj.estimate()

    output_lines = []
    print_report(hw, ds, vram_est, time_res, storage, args, output_lines)

    # ── Export plain text ─────────────────────────────────────────────────────
    if args.export:
        out_path = Path(args.export)
        out_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(GREEN(f"  ✓ Report saved to {out_path.resolve()}"))

    # ── Export JSON ───────────────────────────────────────────────────────────
    if args.json:
        data = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "dataset":    str(ds.path),
                "n_patients": n_eff,
                "patch_size": list(patch),
                "batch_size": args.batch_size,
                "epochs":     args.epochs,
                "n_folds":    args.n_folds,
            },
            "hardware": {
                "gpus":          hw.gpus,
                "cpu_cores":     hw.cpu_cores,
                "cpu_model":     hw.cpu_model,
                "ram_total_gb":  hw.ram_total_gb,
                "ram_avail_gb":  hw.ram_avail_gb,
                "disk_info":     hw.disk_info,
            },
            "dataset": {
                "path":        str(ds.path),
                "exists":      ds.exists,
                "n_patients":  n_eff,
                "raw_size_gb": ds.raw_size_gb,
            },
            "vram_estimates":    vram_est.estimate_all(),
            "time_estimates":    time_res,
            "storage_estimates": storage,
        }
        json_path = Path(args.json)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(GREEN(f"  ✓ JSON data saved to {json_path.resolve()}"))


if __name__ == "__main__":
    main()
