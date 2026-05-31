"""
per_class_analysis.py — Deep per-class AP analysis: ExpA (baseline) vs ExpB (CBAM)

Run from the project root (DSPCBSD/) with the project venv:
    .venv\\Scripts\\python.exe experiments/per_class_analysis.py

Outputs (saved to analysis_outputs/per_class/):
    per_class_metrics.csv     — full table: both experiments × all 9 classes
    ap50_comparison.png       — grouped bar chart AP@0.5
    ap50_95_comparison.png    — grouped bar chart AP@0.5:0.95
    delta_chart.png           — per-class delta (CBAM − expA), sorted
    heatmap.png               — metrics × classes heatmap for both experiments
    radar_chart.png           — spider/radar chart for AP50 per class
    analysis_report.txt       — paper-ready written analysis of results
"""

import re
import sys
import io
import os
import warnings
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CBAM registration (safe even if setup_cbam.py already patched tasks.py)
# ─────────────────────────────────────────────────────────────────────────────

def _register_custom_modules():
    """
    Register all custom attention modules used across experiments:
      • CBAM    — saved under cbam_module.CBAM  (setup_cbam.py registered in tasks.py)
      • CoordAtt — saved under ultralytics.nn.modules.block.CoordAtt
                  (used in [BEST]_ExpB_yolo11s_CA_intial_run_50ep)

    Both must be in-scope BEFORE torch.load() is called, because pickle
    resolves class references by module path at deserialisation time.
    """
    import torch
    import torch.nn as nn

    # ── CBAM ─────────────────────────────────────────────────────────────────
    class ChannelAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            mid = max(channels // reduction, 1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channels, 1, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            mx, _ = torch.max(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    class CBAM(nn.Module):
        def __init__(self, c1: int, c2=None, reduction: int = 16, kernel_size: int = 7):
            super().__init__()
            self.channel_attn = ChannelAttention(c1, reduction)
            self.spatial_attn = SpatialAttention(kernel_size)

        def forward(self, x):
            return self.spatial_attn(self.channel_attn(x))

    # ── CoordAtt (Coordinate Attention — Hou et al., 2021) ───────────────────
    # The [BEST] ExpB checkpoint was trained with this module registered under
    # ultralytics.nn.modules.block.  Pickle will look there at load time.
    class h_sigmoid(nn.Module):
        def __init__(self, inplace: bool = True):
            super().__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace: bool = True):
            super().__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class CoordAtt(nn.Module):
        """Coordinate Attention (Hou et al., CVPR 2021).
        Channel-preserving: output shape == input shape.
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels (must equal c1 — ignored for API compat).
            reduction (int): Channel reduction ratio. Default: 32.
        """
        def __init__(self, c1: int, c2: int = None, reduction: int = 32):
            super().__init__()
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            mip = max(8, c1 // reduction)
            self.conv1  = nn.Conv2d(c1, mip, 1, bias=False)
            self.bn1    = nn.BatchNorm2d(mip)
            self.act    = h_swish()
            self.conv_h = nn.Conv2d(mip, c1, 1, bias=False)
            self.conv_w = nn.Conv2d(mip, c1, 1, bias=False)

        def forward(self, x):
            import torch.nn.functional as F
            n, c, h, w = x.size()
            x_h = F.adaptive_avg_pool2d(x, (h, 1))
            x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
            y   = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
            x_h, x_w = torch.split(y, [h, w], dim=2)
            return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
    
    # ── Register everywhere pickle / parse_model might look ──────────────────
    import types
    import ultralytics.nn.tasks        as _tasks
    import ultralytics.nn.modules.block as _block

    # CBAM → tasks.py globals + cbam_module pseudo-package
    if not hasattr(_tasks, "CBAM"):
        _tasks.CBAM = CBAM
        print("[✓] CBAM registered in ultralytics.nn.tasks")
    else:
        print("[=] CBAM already in ultralytics.nn.tasks")

    cbam_mod = types.ModuleType("cbam_module")
    cbam_mod.CBAM = CBAM
    cbam_mod.ChannelAttention = ChannelAttention # type: ignore
    cbam_mod.SpatialAttention = SpatialAttention # type: ignore
    sys.modules.setdefault("cbam_module", cbam_mod)

    # CoordAtt → ultralytics.nn.modules.block  (where the checkpoint expects it)
    for _cls, _name in [(h_sigmoid, "h_sigmoid"), (h_swish, "h_swish"),
                        (CoordAtt,  "CoordAtt")]:
        if not hasattr(_block, _name):
            setattr(_block, _name, _cls)
            print(f"[✓] {_name} registered in ultralytics.nn.modules.block")
        else:
            print(f"[=] {_name} already in ultralytics.nn.modules.block")

    # Also expose in tasks globals so YAML parse_model can find it
    if not hasattr(_tasks, "CoordAtt"):
        _tasks.CoordAtt = CoordAtt


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths & class metadata
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # DSPCBSD/

# Adjust if your results live elsewhere
_RESULTS_BASE = PROJECT_ROOT.parent / "Claude" / "Projects" / "PCP Defection" / \
                "Experiments_Results_Compined"

EXP0_WEIGHTS = (
    _RESULTS_BASE / "1.Intial_Runs" / "exp0_baseline" /
    "weights" / "best.pt"
)
EXPA_WEIGHTS = (
    _RESULTS_BASE / "2.Based_on_Fixed_Baseline" / "expA_fixed_baseline_100ep" /
    "weights" / "best.pt"
)
CBAM_WEIGHTS = (
    _RESULTS_BASE / "3.CBAM_Runs" / "[BEST]_ExpB_yolo11s_CA_intial_run_50ep" /
    "weights" / "best.pt"
)
DATA_YAML = PROJECT_ROOT / "DSPCBSD+-1" / "data.yaml"

OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "per_class"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DsPCBSD+ class registry (index → abbrev, full name, difficulty)
CLASS_META = [
    (0, "SH",   "Short",                    "medium"),
    (1, "SP",   "Spur",                     "medium"),
    (2, "SC",   "Spurious Copper",          "medium"),
    (3, "OP",   "Open",                     "medium"),
    (4, "MB",   "Mouse Bite",               "medium"),
    (5, "HB",   "Hole Breakout",            "easy"),
    (6, "CS",   "Conductor Scratch",        "hard"),
    (7, "CFO",  "Conductor FOI",            "hard"),
    (8, "BMFO", "Base Material FOI",        "hard"),
]

CLASS_NAMES  = [m[1] for m in CLASS_META]        # ['SH', 'SP', ...]
CLASS_FULL   = [m[2] for m in CLASS_META]
CLASS_DIFF   = [m[3] for m in CLASS_META]

# Shortened names for multi-line plot labels (kept ≤14 chars for readability)
CLASS_SHORT  = [
    "Short",          # SH
    "Spur",           # SP
    "Spur. Copper",   # SC
    "Open",           # OP
    "Mouse Bite",     # MB
    "Hole Breakout",  # HB
    "Cond. Scratch",  # CS
    "Cond. FOI",      # CFO
    "Base Mat. FOI",  # BMFO
]

# Two-line labels: abbreviation bold on top, full name below
DISPLAY_LABELS = [f"{a}\n({s})" for a, s in zip(CLASS_NAMES, CLASS_SHORT)]

DIFF_COLOR  = {"easy": "#27ae60", "medium": "#e67e22", "hard": "#e74c3c"}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Validation runner
# ─────────────────────────────────────────────────────────────────────────────

def _parse_val_table(raw_output: str) -> dict[int, dict]:
    """
    Parse the per-class table that ultralytics prints during val().

    Expected line format (spaces vary):
        [class]  [images]  [instances]  [P]  [R]  [mAP50]  [mAP50-95]
    Class column is a numeric string ('0'..'8') because data.yaml has:
        names: ['0', '1', ..., '8']

    Note: ultralytics uses \\r in tqdm progress bars, which embeds carriage
    returns inside captured lines.  We resolve each line as a terminal would
    (take the last segment after the final \\r) before pattern-matching.
    """
    # Resolve \r as terminal would: take last overwrite segment per line
    clean_lines = []
    for line in raw_output.split("\n"):
        parts = line.split("\r")
        clean_lines.append(parts[-1])   # last \r-segment is what terminal shows
    cleaned = "\n".join(clean_lines)

    # Pattern: word, 4 integers or floats, whitespace-tolerant
    pat = re.compile(
        r"^\s*(\w+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    rows = {}
    for line in cleaned.splitlines():
        m = pat.match(line)
        if not m:
            continue
        cls_str = m.group(1)
        if cls_str in ("all", "Class"):
            continue
        try:
            cls_idx = int(cls_str)
        except ValueError:
            continue
        rows[cls_idx] = {
            "images":     int(m.group(2)),
            "instances":  int(m.group(3)),
            "precision":  float(m.group(4)),
            "recall":     float(m.group(5)),
            "ap50":       float(m.group(6)),
            "ap50_95":    float(m.group(7)),
        }
    return rows


def run_val(weights_path: Path, exp_name: str, device: str = "0") -> dict:
    """
    Load model from weights_path, run val(), return per-class metric dict.

    Returns
    -------
    dict with keys:
        'per_class' : {class_idx: {precision, recall, ap50, ap50_95, f1, instances}}
        'overall'   : {map50, map50_95, precision, recall}
        'raw'       : captured stdout (for debugging)
    """
    print(f"\n{'─'*64}")
    print(f"  Validating: {exp_name}")
    print(f"  Weights   : {weights_path}")
    print(f"{'─'*64}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    import logging
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    # ── Capture ultralytics logger (where the val table is actually printed) ─
    class _LogCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []
        def emit(self, record):
            self.lines.append(self.format(record))

    ul_logger = logging.getLogger("ultralytics")
    handler   = _LogCapture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    ul_logger.addHandler(handler)

    try:
        results = model.val(
            data=str(DATA_YAML),
            imgsz=640,
            batch=8,
            device=device,
            conf=0.001,
            iou=0.6,
            plots=False,
            verbose=True,
            save_json=False,
            save_txt=False,
        )
    finally:
        ul_logger.removeHandler(handler)

    raw = "\n".join(handler.lines)
    print(raw)   # echo captured log to terminal

    # ── Per-class from captured log table ────────────────────────────────
    table = _parse_val_table(raw)

    # ── AP50 / AP50-95 directly from results object (more reliable) ──────
    try:
        ap_class_idx = results.ap_class_index.tolist()
        ap50_arr     = results.box.ap50.tolist()
        ap_arr       = results.box.ap.tolist()
        for rank, ci in enumerate(ap_class_idx):
            if ci not in table:
                table[ci] = {}
            table[ci]["ap50"]    = ap50_arr[rank]
            table[ci]["ap50_95"] = ap_arr[rank]
    except Exception as ex:
        print(f"[!] results.box access failed ({ex}) — using parsed table only")

    # ── P and R per class directly from results (backup if table parse fails) ─
    try:
        ap_class_idx = results.ap_class_index.tolist()
        # results.box.p and results.box.r are per-class arrays at best-F1 threshold
        p_arr = np.array(results.box.p).tolist()
        r_arr = np.array(results.box.r).tolist()
        for rank, ci in enumerate(ap_class_idx):
            if ci not in table:
                table[ci] = {}
            if rank < len(p_arr) and "precision" not in table[ci]:
                table[ci]["precision"] = p_arr[rank]
            if rank < len(r_arr) and "recall" not in table[ci]:
                table[ci]["recall"] = r_arr[rank]
    except Exception as ex:
        print(f"[!] Per-class P/R from results.box failed ({ex})")

    # Compute F1 and attach class meta
    per_class = {}
    for ci in range(9):
        row = table.get(ci, {})
        p = row.get("precision", float("nan"))
        r = row.get("recall",    float("nan"))
        f1 = (2 * p * r / (p + r + 1e-9)) if (not np.isnan(p) and not np.isnan(r)) else float("nan")
        per_class[ci] = {
            "class_abbrev": CLASS_NAMES[ci],
            "class_full":   CLASS_FULL[ci],
            "difficulty":   CLASS_DIFF[ci],
            "instances":    row.get("instances", 0),
            "precision":    p,
            "recall":       r,
            "f1":           f1,
            "ap50":         row.get("ap50",    float("nan")),
            "ap50_95":      row.get("ap50_95", float("nan")),
        }

    # Overall
    try:
        overall = {
            "map50":      results.box.map50,
            "map50_95":   results.box.map,
            "precision":  results.box.mp,
            "recall":     results.box.mr,
        }
    except Exception:
        overall = {k: float("nan") for k in ("map50", "map50_95", "precision", "recall")}

    return {"per_class": per_class, "overall": overall, "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(exp0_data: dict, expa_data: dict, cbam_data: dict) -> pd.DataFrame:
    rows = []
    for ci in range(9):
        e0 = exp0_data["per_class"][ci]
        ea = expa_data["per_class"][ci]
        cb = cbam_data["per_class"][ci]
        rows.append({
            "class_id":    ci,
            "abbrev":      CLASS_NAMES[ci],
            "full_name":   CLASS_FULL[ci],
            "difficulty":  CLASS_DIFF[ci],
            "instances":   ea.get("instances", 0),
            # Exp0
            "exp0_p":      e0["precision"],
            "exp0_r":      e0["recall"],
            "exp0_f1":     e0["f1"],
            "exp0_ap50":   e0["ap50"],
            "exp0_ap50_95":e0["ap50_95"],
            # ExpA
            "expA_p":      ea["precision"],
            "expA_r":      ea["recall"],
            "expA_f1":     ea["f1"],
            "expA_ap50":   ea["ap50"],
            "expA_ap50_95":ea["ap50_95"],
            # CBAM
            "cbam_p":      cb["precision"],
            "cbam_r":      cb["recall"],
            "cbam_f1":     cb["f1"],
            "cbam_ap50":   cb["ap50"],
            "cbam_ap50_95":cb["ap50_95"],
            # Deltas: ExpA vs Exp0 (effect of extra 50 ep)
            "d_expA_ap50":    ea["ap50"]    - e0["ap50"],
            "d_expA_ap50_95": ea["ap50_95"] - e0["ap50_95"],
            "d_expA_f1":      ea["f1"]      - e0["f1"],
            # Deltas: CBAM vs ExpA (effect of attention)
            "d_cbam_ap50":    cb["ap50"]    - ea["ap50"],
            "d_cbam_ap50_95": cb["ap50_95"] - ea["ap50_95"],
            "d_cbam_f1":      cb["f1"]      - ea["f1"],
            # Deltas: CBAM vs Exp0 (total gain)
            "d_total_ap50":    cb["ap50"]    - e0["ap50"],
            "d_total_ap50_95": cb["ap50_95"] - e0["ap50_95"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "exp0": "#8e44ad",   # purple
    "expA": "#2980b9",   # steel blue
    "cbam": "#e74c3c",   # red
    "pos":  "#27ae60",   # green (improvement)
    "neg":  "#c0392b",   # dark red (regression)
    "grid": "#ecf0f1",
}

# Ordered experiment list — used by all multi-experiment plots
EXPS = [
    {"key": "exp0", "label": "Exp0 (50 ep)",      "color": PALETTE["exp0"]},
    {"key": "expA", "label": "ExpA (100 ep)",     "color": PALETTE["expA"]},
    {"key": "cbam", "label": "ExpB CBAM (50 ep)", "color": PALETTE["cbam"]},
]

def _style_ax(ax, title="", ylabel="", ylim=(0, 1), grid=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


# ── 4a. Grouped bar: AP50 ────────────────────────────────────────────────────
def plot_ap50_comparison(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(9)
    w = 0.26
    offsets = [-w, 0, w]

    bars = []
    for exp, off in zip(EXPS, offsets):
        col = f"{exp['key']}_ap50"
        b = ax.bar(x + off, df[col], w, label=exp["label"],
                   color=exp["color"], alpha=0.85, edgecolor="white", linewidth=0.5)
        bars.append(b)
        for bar in b:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8.5)
    for tick, diff in zip(ax.get_xticklabels(), df["difficulty"]):
        tick.set_color(DIFF_COLOR[diff])
        tick.set_fontweight("bold")

    patches = [mpatches.Patch(color=DIFF_COLOR[d], label=d.capitalize())
               for d in ("easy", "medium", "hard")]
    leg1 = ax.legend(handles=patches, title="Difficulty", loc="upper right",
                     fontsize=8, title_fontsize=8, framealpha=0.7)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.add_artist(leg1)

    _style_ax(ax, title="Per-Class AP@0.5 — Exp0 vs. ExpA vs. CBAM", ylabel="AP@0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 4b. Grouped bar: AP50-95 ─────────────────────────────────────────────────
def plot_ap5095_comparison(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(9)
    w = 0.26
    offsets = [-w, 0, w]

    for exp, off in zip(EXPS, offsets):
        col = f"{exp['key']}_ap50_95"
        ax.bar(x + off, df[col], w, label=exp["label"],
               color=exp["color"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8.5)
    ax.legend(fontsize=9)
    _style_ax(ax, title="Per-Class AP@0.5:0.95 — Exp0 vs. ExpA vs. CBAM",
              ylabel="AP@0.5:0.95")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 4c. Delta bar chart — two panels: ExpA−Exp0 and CBAM−ExpA ────────────────
def plot_delta(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    delta_specs = [
        ("d_expA_ap50", "ExpA − Exp0  (extra training, ΔAP@0.5)",
         "ΔAP@0.5", PALETTE["expA"]),
        ("d_cbam_ap50", "CBAM − ExpA  (attention effect, ΔAP@0.5)",
         "ΔAP@0.5", PALETTE["cbam"]),
    ]

    for ax, (col, title, ylabel, base_color) in zip(axes, delta_specs):
        df_s   = df.sort_values(col, ascending=False)
        vals   = df_s[col].values
        labels = df_s["abbrev"].values
        colors = [PALETTE["pos"] if v >= 0 else PALETTE["neg"] for v in vals]

        bars = ax.bar(range(len(vals)), vals, color=colors,
                      edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axhline(0, color="#2c3e50", linewidth=0.9, linestyle="--")

        for bar, v in zip(bars, vals):
            va     = "bottom" if v >= 0 else "top"
            offset = 0.003   if v >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                    f"{v:+.3f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

        sorted_display = [DISPLAY_LABELS[CLASS_NAMES.index(a)] for a in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(sorted_display, fontsize=8.5)

        yabs = max(abs(vals)) if len(vals) > 0 else 0.1
        _style_ax(ax, title=title, ylabel=ylabel, ylim=(-yabs * 1.4, yabs * 1.4))

        for tick, abbrev in zip(ax.get_xticklabels(), labels):
            ci = CLASS_NAMES.index(abbrev)
            tick.set_color(DIFF_COLOR[CLASS_DIFF[ci]])
            tick.set_fontweight("bold")

    for ax in axes:
        ax.legend(handles=[
            mpatches.Patch(color=PALETTE["pos"], label="Improves"),
            mpatches.Patch(color=PALETTE["neg"], label="Degrades"),
        ], fontsize=8, loc="lower right")

    fig.suptitle("Per-Class AP@0.5 Deltas Across Experiments",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 4d. Heatmap ──────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, path: Path):
    metrics = ["ap50", "ap50_95", "f1", "p", "r"]
    metric_labels = ["AP@0.5", "AP@0.5:0.95", "F1", "Precision", "Recall"]

    # Build (3*metrics) × classes matrix — one block per experiment
    rows_e0 = [df[f"exp0_{m}"].values for m in metrics]
    rows_ea = [df[f"expA_{m}"].values for m in metrics]
    rows_cb = [df[f"cbam_{m}"].values for m in metrics]
    data = np.array(rows_e0 + rows_ea + rows_cb, dtype=float)

    row_labels = [f"Exp0  – {ml}" for ml in metric_labels] + \
                 [f"ExpA  – {ml}" for ml in metric_labels] + \
                 [f"CBAM – {ml}"  for ml in metric_labels]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(9))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Separators between experiment blocks
    ax.axhline(len(metrics) - 0.5, color="white", linewidth=2.5)
    ax.axhline(2 * len(metrics) - 0.5, color="white", linewidth=2.5)

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "–"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color="black" if 0.25 < v < 0.85 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Metric value")
    ax.set_title("Per-Class Metric Heatmap — ExpA vs. CBAM", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 4e. Radar chart ──────────────────────────────────────────────────────────
def plot_radar(df: pd.DataFrame, path: Path):
    categories = [f"{a}\n({s})" for a, s in zip(CLASS_NAMES, CLASS_SHORT)]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def _vals(col):
        v = df[col].fillna(0).tolist()
        v += v[:1]
        return v

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for exp in EXPS:
        vals = _vals(f"{exp['key']}_ap50")
        ax.plot(angles, vals, color=exp["color"], linewidth=2,
                linestyle="solid", label=exp["label"])
        ax.fill(angles, vals, color=exp["color"], alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=7, color="grey")
    ax.set_title("Per-Class AP@0.5 Radar — Exp0 vs. ExpA vs. CBAM",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 4f. Precision–Recall scatter per class ───────────────────────────────────
def plot_pr_scatter(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, exp in zip(axes, EXPS):
        exp_prefix = exp["key"]
        title      = exp["label"]
        for _, row in df.iterrows():
            p = row[f"{exp_prefix}_p"]
            r = row[f"{exp_prefix}_r"]
            color = DIFF_COLOR[row["difficulty"]]
            ax.scatter(r, p, s=120, color=color, zorder=5, edgecolors="white", linewidths=0.8)
            short = CLASS_SHORT[CLASS_NAMES.index(row["abbrev"])]
            ax.annotate(f"{row['abbrev']}\n({short})", (r, p),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7.5, fontweight="bold", color=color)

        # F1 iso-curves
        recall_range = np.linspace(0.01, 1.0, 200)
        for f1_val in [0.3, 0.5, 0.7, 0.9]:
            prec_line = f1_val * recall_range / (2 * recall_range - f1_val + 1e-9)
            mask = (prec_line >= 0) & (prec_line <= 1)
            ax.plot(recall_range[mask], prec_line[mask], color="#bdc3c7",
                    linewidth=0.8, linestyle="--")
            idx = np.argmin(np.abs(recall_range - 0.9))
            if mask[idx]:
                ax.text(recall_range[idx], prec_line[idx], f"F1={f1_val}",
                        fontsize=6.5, color="#95a5a6")

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        _style_ax(ax, title=title, ylim=None)

    # Difficulty legend
    patches = [mpatches.Patch(color=DIFF_COLOR[d], label=d.capitalize())
               for d in ("easy", "medium", "hard")]
    fig.legend(handles=patches, title="Difficulty", loc="lower center",
               ncol=3, fontsize=9, title_fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Precision vs. Recall per Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Analysis report (paper-ready text)
# ─────────────────────────────────────────────────────────────────────────────

def write_analysis_report(df: pd.DataFrame, expa_overall: dict, cbam_overall: dict,
                          path: Path):
    lines = []
    _w = lines.append

    _w("=" * 76)
    _w("  PER-CLASS ANALYSIS REPORT — Exp0 | ExpA | ExpB (CBAM)")
    _w("=" * 76)
    _w("")

    # ── Overall metrics ───────────────────────────────────────────────────
    _w("── OVERALL METRICS (ExpA vs CBAM) ───────────────────────────────────")
    _w(f"{'Metric':<22}  {'ExpA':>8}  {'CBAM':>8}  {'Δ CBAM−ExpA':>12}")
    _w("-" * 58)
    for key, label in [("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95"),
                        ("precision", "Precision"), ("recall", "Recall")]:
        ea_v  = expa_overall.get(key, float("nan"))
        cb_v  = cbam_overall.get(key, float("nan"))
        delta = cb_v - ea_v
        _w(f"  {label:<20}  {ea_v:>8.4f}  {cb_v:>8.4f}  {delta:>+12.4f}")
    _w("")

    # ── Per-class AP50 full table ─────────────────────────────────────────
    _w("── PER-CLASS AP@0.5 TABLE ───────────────────────────────────────────")
    _w(f"{'Class':<18}  {'Diff':<6}  {'Exp0':>7}  {'ExpA':>7}  {'CBAM':>7}  "
       f"{'Δ ExpA−Exp0':>12}  {'Δ CBAM−ExpA':>12}")
    _w("-" * 82)
    for _, row in df.iterrows():
        _w(f"  {row['full_name']:<16}  {row['difficulty']:<6}  "
           f"{row['exp0_ap50']:>7.4f}  {row['expA_ap50']:>7.4f}  {row['cbam_ap50']:>7.4f}  "
           f"{row['d_expA_ap50']:>+12.4f}  {row['d_cbam_ap50']:>+12.4f}")
    _w("")

    # ── CBAM vs ExpA: winners / losers ────────────────────────────────────
    improved  = df[df["d_cbam_ap50"] >  0.005].sort_values("d_cbam_ap50", ascending=False)
    degraded  = df[df["d_cbam_ap50"] < -0.005].sort_values("d_cbam_ap50")
    unchanged = df[df["d_cbam_ap50"].abs() <= 0.005]

    _w("── CBAM vs ExpA — WINNERS (Δ > +0.5 pp) ────────────────────────────")
    if len(improved):
        for _, row in improved.iterrows():
            _w(f"  {row['abbrev']:>4} ({row['full_name']:<22})  "
               f"Δ={row['d_cbam_ap50']:+.4f}  "
               f"[{row['expA_ap50']:.4f} → {row['cbam_ap50']:.4f}]  "
               f"difficulty={row['difficulty']}")
    else:
        _w("  None — CBAM did not improve any class by more than 0.5 pp.")
    _w("")

    _w("── CBAM vs ExpA — LOSERS (Δ < −0.5 pp) ────────────────────────────")
    if len(degraded):
        for _, row in degraded.iterrows():
            _w(f"  {row['abbrev']:>4} ({row['full_name']:<22})  "
               f"Δ={row['d_cbam_ap50']:+.4f}  "
               f"[{row['expA_ap50']:.4f} → {row['cbam_ap50']:.4f}]  "
               f"difficulty={row['difficulty']}")
    else:
        _w("  None.")
    _w("")

    _w("── CBAM vs ExpA — STABLE (|Δ| ≤ 0.5 pp) ───────────────────────────")
    if len(unchanged):
        for _, row in unchanged.iterrows():
            _w(f"  {row['abbrev']:>4} ({row['full_name']:<22})  "
               f"Δ={row['d_cbam_ap50']:+.4f}  ExpA AP50={row['expA_ap50']:.4f}")
    else:
        _w("  None.")
    _w("")

    # ── ExpA vs Exp0: effect of extra training ────────────────────────────
    _w("── EXTRA TRAINING EFFECT (ExpA − Exp0, +50 epochs) ─────────────────")
    _w(f"{'Class':<18}  {'Δ AP50':>8}  {'Δ AP50-95':>10}")
    _w("-" * 42)
    for _, row in df.sort_values("d_expA_ap50", ascending=False).iterrows():
        _w(f"  {row['full_name']:<16}  {row['d_expA_ap50']:>+8.4f}  "
           f"{row['d_expA_ap50_95']:>+10.4f}")
    _w("")

    # ── Delta by difficulty group ─────────────────────────────────────────
    _w("── MEAN DELTA BY DIFFICULTY GROUP ───────────────────────────────────")
    _w(f"{'Group':<8}  {'Classes':<30}  {'Δ ExpA−Exp0':>12}  {'Δ CBAM−ExpA':>12}")
    _w("-" * 68)
    for diff in ("easy", "medium", "hard"):
        sub     = df[df["difficulty"] == diff]
        classes = ", ".join(sub["abbrev"].tolist())
        d_train = sub["d_expA_ap50"].mean()
        d_cbam  = sub["d_cbam_ap50"].mean()
        _w(f"  {diff.capitalize():<6}  {classes:<30}  {d_train:>+12.4f}  {d_cbam:>+12.4f}")
    _w("")

    # ── P / R / F1 shift (CBAM vs ExpA) ──────────────────────────────────
    _w("── PRECISION / RECALL / F1 SHIFT (CBAM − ExpA) ─────────────────────")
    _w(f"{'Class':<18}  {'Δ Precision':>12}  {'Δ Recall':>10}  {'Δ F1':>8}")
    _w("-" * 56)
    for _, row in df.iterrows():
        dp = row["cbam_p"] - row["expA_p"]
        dr = row["cbam_r"] - row["expA_r"]
        df1 = row["cbam_f1"] - row["expA_f1"]
        _w(f"  {row['full_name']:<16}  {dp:>+12.4f}  {dr:>+10.4f}  {df1:>+8.4f}")
    _w("")

    # ── Paper interpretation ──────────────────────────────────────────────
    _w("── INTERPRETATION (for paper) ───────────────────────────────────────")
    net = cbam_overall.get("map50", float("nan")) - expa_overall.get("map50", float("nan"))
    _w(f"  CBAM (CoordAtt) changed overall mAP@0.5 vs ExpA by {net:+.4f}.")
    _w(f"  Class-level: {len(improved)} improved, {len(degraded)} degraded, "
       f"{len(unchanged)} stable (threshold ±0.5 pp).")

    for diff in ("easy", "medium", "hard"):
        sub    = df[df["difficulty"] == diff]
        d_cbam = sub["d_cbam_ap50"].mean()
        cls    = "/".join(sub["abbrev"].tolist())
        _w(f"  {diff.capitalize()} ({cls}): mean CBAM Δ = {d_cbam:+.4f}")

    total_hard = df[df["difficulty"] == "hard"]["d_total_ap50"].mean()
    _w(f"  Hard classes total gain vs Exp0: {total_hard:+.4f} "
       f"(training + attention combined).")
    _w("")
    _w("=" * 76)

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    print(f"  [✓] {path.name}")
    print()
    print(report_text)
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 64)
    print("  PCB Defect Detection — Per-Class Analysis")
    print("  Exp0 (50 ep) | ExpA (100 ep) | ExpB CBAM (50 ep)")
    print("=" * 64)

    # ── Register custom attention modules (CBAM + CoordAtt) ──────────────
    _register_custom_modules()

    device = "0" if __import__("torch").cuda.is_available() else "cpu"
    print(f"\n  Running on device: {device}")

    # ── Validate all three models ─────────────────────────────────────────
    print("\n[STEP 1/4] Running Exp0 validation ...")
    exp0 = run_val(EXP0_WEIGHTS, "Exp0 — Initial Baseline (50 ep)", device=device)

    print("\n[STEP 2/4] Running ExpA validation ...")
    expa = run_val(EXPA_WEIGHTS, "ExpA — Fixed Baseline (100 ep)", device=device)

    print("\n[STEP 3/4] Running CBAM (ExpB) validation ...")
    cbam = run_val(CBAM_WEIGHTS, "ExpB — CBAM Attention (50 ep)", device=device)

    # ── Build combined DataFrame ──────────────────────────────────────────
    print("\n[STEP 4/4] Building table & generating plots ...")
    df = build_dataframe(exp0, expa, cbam)

    # Save CSV
    csv_path = OUT_DIR / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  [✓] {csv_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_ap50_comparison (df, OUT_DIR / "ap50_comparison.png")
    plot_ap5095_comparison(df, OUT_DIR / "ap50_95_comparison.png")
    plot_delta           (df, OUT_DIR / "delta_chart.png")
    plot_heatmap         (df, OUT_DIR / "heatmap.png")
    plot_radar           (df, OUT_DIR / "radar_chart.png")
    plot_pr_scatter      (df, OUT_DIR / "pr_scatter.png")

    # ── Analysis report ───────────────────────────────────────────────────
    write_analysis_report(df, expa["overall"], cbam["overall"],
                          OUT_DIR / "analysis_report.txt")

    print(f"\n{'='*64}")
    print(f"  All outputs saved to: {OUT_DIR}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
