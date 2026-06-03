"""
per_class_analysis.py — Per-class AP analysis across all presentation experiments

Run from DSPCBSD/ with:
    .venv\\Scripts\\python.exe experiments/per_class_analysis.py

Experiments validated (all have weights/best.pt in "Experiments to Present"):
    1.1  Baseline 50ep
    1.2  Baseline 50ep + Advanced Augmentation
    2.1  Baseline 100ep
    2.2  Baseline 100ep + Oversample
    3.   CA (CoordAtt)
    4.   CBAM
    5.   P2

Outputs saved to analysis_outputs/per_class/:
    per_class_metrics.csv
    ap50_comparison.png
    ap50_95_comparison.png
    delta_chart.png
    heatmap.png
    radar_chart.png
    pr_scatter.png
    analysis_report.txt
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Custom module registration  (CBAM + CoordAtt)
# ─────────────────────────────────────────────────────────────────────────────

def _register_custom_modules():
    import torch
    import torch.nn as nn
    import types
    import ultralytics.nn.tasks        as _tasks
    import ultralytics.nn.modules.block as _block

    # ── CBAM ─────────────────────────────────────────────────────────────────
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
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
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            mx, _ = torch.max(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    class CBAM(nn.Module):
        def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
            super().__init__()
            self.channel_attn = ChannelAttention(c1, reduction)
            self.spatial_attn = SpatialAttention(kernel_size)
        def forward(self, x):
            return self.spatial_attn(self.channel_attn(x))

    # ── CoordAtt ─────────────────────────────────────────────────────────────
    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.relu = nn.ReLU6(inplace=inplace)
        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)
        def forward(self, x):
            return x * self.sigmoid(x)

    class CoordAtt(nn.Module):
        def __init__(self, c1, c2=None, reduction=32):
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

    # ── Register ─────────────────────────────────────────────────────────────
    if not hasattr(_tasks, "CBAM"):
        _tasks.CBAM = CBAM
        print("[✓] CBAM registered in ultralytics.nn.tasks")
    else:
        print("[=] CBAM already registered")

    cbam_mod = types.ModuleType("cbam_module")
    cbam_mod.CBAM             = CBAM
    cbam_mod.ChannelAttention = ChannelAttention
    cbam_mod.SpatialAttention = SpatialAttention
    sys.modules.setdefault("cbam_module", cbam_mod)

    for _cls, _name in [(h_sigmoid, "h_sigmoid"), (h_swish, "h_swish"), (CoordAtt, "CoordAtt")]:
        if not hasattr(_block, _name):
            setattr(_block, _name, _cls)
            print(f"[✓] {_name} registered in ultralytics.nn.modules.block")
        else:
            print(f"[=] {_name} already registered")

    if not hasattr(_tasks, "CoordAtt"):
        _tasks.CoordAtt = CoordAtt


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths & experiment registry
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # DSPCBSD/

BASE_DIR = Path(
    r"C:\Users\Yasmi\OneDrive\Desktop\UNI\This semester\[Projects]\Machine II"
    r"\[Presentation_Preperation]\Experiments to Present"
)

DATA_YAML = PROJECT_ROOT / "DSPCBSD+-1" / "data.yaml"
OUT_DIR   = PROJECT_ROOT / "analysis_outputs" / "per_class"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: key (short id), label (plot label), color, weights path.
EXPERIMENTS = [
    {
        "key":     "base50",
        "label":   "1.1 Baseline\n50ep",
        "color":   "#95a5a6",
        "weights": BASE_DIR / "1.Baseline_50" / "1.1.Baseline_50" / "weights" / "best.pt",
    },
    {
        "key":     "base50aug",
        "label":   "1.2 Baseline\n50ep+Aug",
        "color":   "#7f8c8d",
        "weights": BASE_DIR / "1.Baseline_50" / "1.2.Baseline_50+Advanced_Augmentation" / "weights" / "best.pt",
    },
    {
        "key":     "base100",
        "label":   "2.1 Baseline\n100ep",
        "color":   "#8e44ad",
        "weights": BASE_DIR / "2.Baseline_100" / "2.1.Baseline_100" / "weights" / "best.pt",
    },
    {
        "key":     "base100os",
        "label":   "2.2 Baseline\n+Oversample",
        "color":   "#2980b9",
        "weights": BASE_DIR / "2.Baseline_100" / "2.2.Baseline_100+oversample" / "weights" / "best.pt",
    },
    {
        "key":     "ca",
        "label":   "3. CA\n(CoordAtt)",
        "color":   "#27ae60",
        "weights": BASE_DIR / "3.CA" / "weights" / "best.pt",
    },
    {
        "key":     "cbam",
        "label":   "4. CBAM",
        "color":   "#e67e22",
        "weights": BASE_DIR / "4.CBAM" / "weights" / "best.pt",
    },
    {
        "key":     "p2",
        "label":   "5. P2",
        "color":   "#e74c3c",
        "weights": BASE_DIR / "5.P2" / "weights" / "best.pt",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Class metadata
# ─────────────────────────────────────────────────────────────────────────────

CLASS_META = [
    (0, "SH",   "Short",             "medium"),
    (1, "SP",   "Spur",              "medium"),
    (2, "SC",   "Spurious Copper",   "medium"),
    (3, "OP",   "Open",              "medium"),
    (4, "MB",   "Mouse Bite",        "medium"),
    (5, "HB",   "Hole Breakout",     "easy"),
    (6, "CS",   "Conductor Scratch", "hard"),
    (7, "CFO",  "Conductor FOI",     "hard"),
    (8, "BMFO", "Base Material FOI", "hard"),
]

CLASS_NAMES  = [m[1] for m in CLASS_META]
CLASS_FULL   = [m[2] for m in CLASS_META]
CLASS_DIFF   = [m[3] for m in CLASS_META]
CLASS_SHORT  = [
    "Short", "Spur", "Spur. Copper", "Open", "Mouse Bite",
    "Hole Breakout", "Cond. Scratch", "Cond. FOI", "Base Mat. FOI",
]
DISPLAY_LABELS = [f"{a}\n({s})" for a, s in zip(CLASS_NAMES, CLASS_SHORT)]
DIFF_COLOR     = {"easy": "#27ae60", "medium": "#e67e22", "hard": "#e74c3c"}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Validation runner
# ─────────────────────────────────────────────────────────────────────────────

def _parse_val_table(raw_output: str) -> dict:
    clean_lines = []
    for line in raw_output.split("\n"):
        parts = line.split("\r")
        clean_lines.append(parts[-1])
    cleaned = "\n".join(clean_lines)

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
            "images":    int(m.group(2)),
            "instances": int(m.group(3)),
            "precision": float(m.group(4)),
            "recall":    float(m.group(5)),
            "ap50":      float(m.group(6)),
            "ap50_95":   float(m.group(7)),
        }
    return rows


def run_val(weights_path: Path, exp_label: str, device: str = "0") -> dict:
    print(f"\n{'─'*64}")
    print(f"  Validating : {exp_label}")
    print(f"  Weights    : {weights_path}")
    print(f"{'─'*64}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    import logging
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    class _LogCapture(logging.Handler):
        def __init__(self): super().__init__(); self.lines = []
        def emit(self, record): self.lines.append(self.format(record))

    ul_logger = logging.getLogger("ultralytics")
    handler   = _LogCapture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    ul_logger.addHandler(handler)

    try:
        results = model.val(
            data=str(DATA_YAML),
            imgsz=640, batch=8, device=device,
            conf=0.001, iou=0.6,
            plots=False, verbose=True,
            save_json=False, save_txt=False,
        )
    finally:
        ul_logger.removeHandler(handler)

    raw = "\n".join(handler.lines)
    print(raw)

    table = _parse_val_table(raw)

    # Override AP from results object (more reliable than parsed table)
    try:
        ap_class_idx = results.ap_class_index.tolist()
        ap50_arr     = results.box.ap50.tolist()
        ap_arr       = results.box.ap.tolist()
        for rank, ci in enumerate(ap_class_idx):
            if ci not in table: table[ci] = {}
            table[ci]["ap50"]    = ap50_arr[rank]
            table[ci]["ap50_95"] = ap_arr[rank]
    except Exception as ex:
        print(f"[!] results.box AP access failed ({ex})")

    try:
        ap_class_idx = results.ap_class_index.tolist()
        p_arr = np.array(results.box.p).tolist()
        r_arr = np.array(results.box.r).tolist()
        for rank, ci in enumerate(ap_class_idx):
            if ci not in table: table[ci] = {}
            if rank < len(p_arr) and "precision" not in table[ci]:
                table[ci]["precision"] = p_arr[rank]
            if rank < len(r_arr) and "recall" not in table[ci]:
                table[ci]["recall"] = r_arr[rank]
    except Exception as ex:
        print(f"[!] Per-class P/R failed ({ex})")

    per_class = {}
    for ci in range(9):
        row = table.get(ci, {})
        p  = row.get("precision", float("nan"))
        r  = row.get("recall",    float("nan"))
        f1 = (2*p*r / (p+r+1e-9)) if (not np.isnan(p) and not np.isnan(r)) else float("nan")
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

    try:
        overall = {
            "map50":     results.box.map50,
            "map50_95":  results.box.map,
            "precision": results.box.mp,
            "recall":    results.box.mr,
        }
    except Exception:
        overall = {k: float("nan") for k in ("map50", "map50_95", "precision", "recall")}

    return {"per_class": per_class, "overall": overall, "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DataFrame builder — N experiments
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(exp_results: list) -> pd.DataFrame:
    """
    exp_results: list of dicts with keys 'key', 'label', 'data'
    where 'data' is the return value of run_val().
    Produces one row per class, columns: {key}_p/r/f1/ap50/ap50_95 per experiment.
    """
    rows = []
    for ci in range(9):
        row = {
            "class_id":   ci,
            "abbrev":     CLASS_NAMES[ci],
            "full_name":  CLASS_FULL[ci],
            "difficulty": CLASS_DIFF[ci],
            "instances":  exp_results[0]["data"]["per_class"][ci].get("instances", 0),
        }
        for exp in exp_results:
            key = exp["key"]
            pc  = exp["data"]["per_class"][ci]
            row[f"{key}_p"]       = pc["precision"]
            row[f"{key}_r"]       = pc["recall"]
            row[f"{key}_f1"]      = pc["f1"]
            row[f"{key}_ap50"]    = pc["ap50"]
            row[f"{key}_ap50_95"] = pc["ap50_95"]
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_GRID_COLOR = "#ecf0f1"


def _style_ax(ax, title="", ylabel="", ylim=(0, 1), grid=True):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)


# ── 5a. Grouped bar: AP50 ────────────────────────────────────────────────────
def plot_ap50_comparison(df: pd.DataFrame, exp_list: list, path: Path):
    n     = len(exp_list)
    total = 0.8
    w     = total / n
    x     = np.arange(9)
    offs  = np.linspace(-(total - w) / 2, (total - w) / 2, n)

    fig, ax = plt.subplots(figsize=(16, 5))
    for exp, off in zip(exp_list, offs):
        b = ax.bar(x + off, df[f"{exp['key']}_ap50"], w,
                   label=exp["label"].replace("\n", " "),
                   color=exp["color"], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar in b:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8)
    for tick, diff in zip(ax.get_xticklabels(), df["difficulty"]):
        tick.set_color(DIFF_COLOR[diff])
        tick.set_fontweight("bold")

    diff_patches = [mpatches.Patch(color=DIFF_COLOR[d], label=d.capitalize())
                    for d in ("easy", "medium", "hard")]
    leg_diff = ax.legend(handles=diff_patches, title="Difficulty",
                         loc="upper right", fontsize=8, title_fontsize=8, framealpha=0.7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7, ncol=2)
    ax.add_artist(leg_diff)

    _style_ax(ax, title="Per-Class AP@0.5 — All Experiments", ylabel="AP@0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 5b. Grouped bar: AP50-95 ─────────────────────────────────────────────────
def plot_ap5095_comparison(df: pd.DataFrame, exp_list: list, path: Path):
    n     = len(exp_list)
    total = 0.8
    w     = total / n
    x     = np.arange(9)
    offs  = np.linspace(-(total - w) / 2, (total - w) / 2, n)

    fig, ax = plt.subplots(figsize=(16, 5))
    for exp, off in zip(exp_list, offs):
        ax.bar(x + off, df[f"{exp['key']}_ap50_95"], w,
               label=exp["label"].replace("\n", " "),
               color=exp["color"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8)
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax, title="Per-Class AP@0.5:0.95 — All Experiments", ylabel="AP@0.5:0.95")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 5c. Delta bars — each experiment vs. baseline (first in list) ─────────────
def plot_delta(df: pd.DataFrame, exp_list: list, path: Path):
    baseline_key = exp_list[0]["key"]
    others       = exp_list[1:]
    n            = len(others)
    n_cols       = 2
    n_rows       = (n + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, exp in enumerate(others):
        ax        = axes[i]
        delta_col = f"d_{exp['key']}_ap50"
        df[delta_col] = df[f"{exp['key']}_ap50"] - df[f"{baseline_key}_ap50"]

        df_s   = df.sort_values(delta_col, ascending=False)
        vals   = df_s[delta_col].values
        labels = df_s["abbrev"].values
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]

        bars = ax.bar(range(len(vals)), vals, color=colors,
                      edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axhline(0, color="#2c3e50", linewidth=0.9, linestyle="--")

        for bar, v in zip(bars, vals):
            va     = "bottom" if v >= 0 else "top"
            offset = 0.003    if v >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:+.3f}", ha="center", va=va, fontsize=7, fontweight="bold")

        sorted_disp = [DISPLAY_LABELS[CLASS_NAMES.index(a)] for a in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(sorted_disp, fontsize=7.5)
        yabs = max(abs(vals)) if len(vals) else 0.1
        _style_ax(ax,
                  title=f"{exp['label'].replace(chr(10),' ')} − Baseline 100ep  (ΔAP@0.5)",
                  ylabel="ΔAP@0.5",
                  ylim=(-yabs * 1.4, yabs * 1.4))

        for tick, abbrev in zip(ax.get_xticklabels(), labels):
            ci = CLASS_NAMES.index(abbrev)
            tick.set_color(DIFF_COLOR[CLASS_DIFF[ci]])
            tick.set_fontweight("bold")

        ax.legend(handles=[
            mpatches.Patch(color="#27ae60", label="Improves"),
            mpatches.Patch(color="#c0392b", label="Degrades"),
        ], fontsize=8, loc="lower right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Class ΔAP@0.5 vs Baseline 100ep",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 5d. Heatmap ──────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, exp_list: list, path: Path):
    metrics       = ["ap50", "ap50_95", "f1", "p", "r"]
    metric_labels = ["AP@0.5", "AP@0.5:0.95", "F1", "Precision", "Recall"]

    all_rows    = []
    row_labels  = []
    for exp in exp_list:
        for m, ml in zip(metrics, metric_labels):
            col = f"{exp['key']}_{m}"
            all_rows.append(df[col].values)
            row_labels.append(f"{exp['label'].replace(chr(10),' ')} – {ml}")

    data   = np.array(all_rows, dtype=float)
    n_rows = len(row_labels)

    fig, ax = plt.subplots(figsize=(14, max(8, n_rows * 0.52)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(9))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=8, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    # White separators between experiment blocks
    for i in range(1, len(exp_list)):
        ax.axhline(i * len(metrics) - 0.5, color="white", linewidth=2.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v   = data[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "–"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="black" if 0.25 < v < 0.85 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label="Metric value")
    ax.set_title("Per-Class Metric Heatmap — All Experiments",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 5e. Radar chart ──────────────────────────────────────────────────────────
def plot_radar(df: pd.DataFrame, exp_list: list, path: Path):
    N      = 9
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for exp in exp_list:
        vals = df[f"{exp['key']}_ap50"].fillna(0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=exp["color"], linewidth=2,
                label=exp["label"].replace("\n", " "))
        ax.fill(angles, vals, color=exp["color"], alpha=0.07)

    categories = [f"{a}\n({s})" for a, s in zip(CLASS_NAMES, CLASS_SHORT)]
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=7, color="grey")
    ax.set_title("Per-Class AP@0.5 Radar — All Experiments",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── 5f. Precision–Recall scatter ─────────────────────────────────────────────
def plot_pr_scatter(df: pd.DataFrame, exp_list: list, path: Path):
    n    = len(exp_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    recall_range = np.linspace(0.01, 1.0, 200)
    for ax, exp in zip(axes, exp_list):
        for _, row in df.iterrows():
            p     = row[f"{exp['key']}_p"]
            r     = row[f"{exp['key']}_r"]
            color = DIFF_COLOR[row["difficulty"]]
            ax.scatter(r, p, s=110, color=color, zorder=5,
                       edgecolors="white", linewidths=0.8)
            short = CLASS_SHORT[CLASS_NAMES.index(row["abbrev"])]
            ax.annotate(f"{row['abbrev']}\n({short})", (r, p),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, fontweight="bold", color=color)

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
        _style_ax(ax, title=exp["label"].replace("\n", " "), ylim=None)

    diff_patches = [mpatches.Patch(color=DIFF_COLOR[d], label=d.capitalize())
                    for d in ("easy", "medium", "hard")]
    fig.legend(handles=diff_patches, title="Difficulty", loc="lower center",
               ncol=3, fontsize=9, title_fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Precision vs. Recall per Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Analysis report
# ─────────────────────────────────────────────────────────────────────────────

def write_analysis_report(df: pd.DataFrame, exp_results: list, path: Path):
    lines = []
    _w = lines.append

    _w("=" * 76)
    _w("  PER-CLASS ANALYSIS REPORT — Experiments to Present")
    _w("=" * 76)
    _w("")

    # Overall metrics
    _w("── OVERALL METRICS ──────────────────────────────────────────────────")
    _w(f"{'Experiment':<28}  {'mAP@0.5':>8}  {'mAP@0.5:0.95':>13}  {'P':>8}  {'R':>8}")
    _w("-" * 74)
    for exp in exp_results:
        ov    = exp["data"]["overall"]
        label = exp["label"].replace("\n", " ")
        _w(f"  {label:<26}  {ov['map50']:>8.4f}  "
           f"{ov['map50_95']:>13.4f}  {ov['precision']:>8.4f}  {ov['recall']:>8.4f}")
    _w("")

    # Per-class AP50 table
    _w("── PER-CLASS AP@0.5 TABLE ───────────────────────────────────────────")
    header = f"{'Class':<18}  {'Diff':<6}" + "".join(
        f"  {exp['label'].replace(chr(10),' ')[:13]:>13}" for exp in exp_results
    )
    _w(header)
    _w("-" * (26 + 15 * len(exp_results)))
    for _, row in df.iterrows():
        line = f"  {row['full_name']:<16}  {row['difficulty']:<6}"
        for exp in exp_results:
            k = exp["key"]
            line += f"  {row[k + '_ap50']:>13.4f}"
        _w(line)
    _w("")

    # Deltas vs baseline
    baseline_key = exp_results[0]["key"]
    _w("── DELTA vs BASELINE (AP@0.5) ───────────────────────────────────────")
    for exp in exp_results[1:]:
        k     = exp["key"]
        col   = f"d_{k}_ap50"
        label = exp["label"].replace("\n", " ")
        if col not in df.columns:
            df[col] = df[k + "_ap50"] - df[baseline_key + "_ap50"]
        improved = df[df[col] >  0.005]
        degraded = df[df[col] < -0.005]
        _w(f"  {label}: {len(improved)} improved, {len(degraded)} degraded vs baseline")
        if len(df):
            best_cls  = df.loc[df[col].idxmax(), "full_name"]
            worst_cls = df.loc[df[col].idxmin(), "full_name"]
            _w(f"    Best gain : {best_cls} ({df[col].max():+.4f})")
            _w(f"    Worst loss: {worst_cls} ({df[col].min():+.4f})")
    _w("")

    # Difficulty breakdown
    _w("── MEAN AP@0.5 BY DIFFICULTY GROUP ─────────────────────────────────")
    _w(f"{'Group':<8}  {'Classes':<30}" + "".join(
        f"  {exp['label'].replace(chr(10),' ')[:10]:>10}" for exp in exp_results
    ))
    _w("-" * (40 + 12 * len(exp_results)))
    for diff in ("easy", "medium", "hard"):
        sub     = df[df["difficulty"] == diff]
        classes = ", ".join(sub["abbrev"].tolist())
        line    = f"  {diff.capitalize():<6}  {classes:<30}"
        for exp in exp_results:
            k = exp["key"]
            line += f"  {sub[k + '_ap50'].mean():>10.4f}"
        _w(line)
    _w("")
    _w("=" * 76)

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    print(f"  [✓] {path.name}")
    print()
    print(report_text)
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 64)
    print("  PCB Defect Detection — Per-Class Analysis")
    print("  Experiments to Present  (7 runs)")
    print("=" * 64)

    _register_custom_modules()

    device = "0" if __import__("torch").cuda.is_available() else "cpu"
    print(f"\n  Running on device: {device}")

    # ── Validate all experiments ──────────────────────────────────────────
    exp_results = []
    total = len(EXPERIMENTS)
    for i, exp in enumerate(EXPERIMENTS, 1):
        label = exp["label"].replace("\n", " ")
        print(f"\n[STEP {i}/{total}] {label}")
        data = run_val(exp["weights"], label, device=device)
        exp_results.append({"key": exp["key"], "label": exp["label"], "data": data})

    # ── Build DataFrame ───────────────────────────────────────────────────
    step = total + 1
    print(f"\n[STEP {step}/{step}] Building table & generating plots ...")
    df = build_dataframe(exp_results)

    csv_path = OUT_DIR / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  [✓] {csv_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_ap50_comparison (df, EXPERIMENTS, OUT_DIR / "ap50_comparison.png")
    plot_ap5095_comparison(df, EXPERIMENTS, OUT_DIR / "ap50_95_comparison.png")
    plot_delta           (df, EXPERIMENTS, OUT_DIR / "delta_chart.png")
    plot_heatmap         (df, EXPERIMENTS, OUT_DIR / "heatmap.png")
    plot_radar           (df, EXPERIMENTS, OUT_DIR / "radar_chart.png")
    plot_pr_scatter      (df, EXPERIMENTS, OUT_DIR / "pr_scatter.png")

    # ── Report ────────────────────────────────────────────────────────────
    write_analysis_report(df, exp_results, OUT_DIR / "analysis_report.txt")

    print(f"\n{'='*64}")
    print(f"  All outputs saved to: {OUT_DIR}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
