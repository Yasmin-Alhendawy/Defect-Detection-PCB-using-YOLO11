"""
PCB Defect Detection — Regenerate Training Plots & Confusion Matrices
======================================================================
Edit the 3 variables in the CONFIG section, then run:

    python regen_plots_and_cm.py --plots        # training curves only (no GPU)
    python regen_plots_and_cm.py --cm           # confusion matrices only (GPU)
    python regen_plots_and_cm.py --plots --cm   # both
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit these 3 lines for each experiment
# ══════════════════════════════════════════════════════════════════════════════

EXP_FOLDER = Path(r"C:\Users\Yasmi\OneDrive\Desktop\UNI\This semester\[Projects]\Machine II\[Presentation_Preperation]\Experiments to Present\1.Baseline_50\2.Baseline_50+Advanced_Augmentation")
EXP_NAME   = "Baseline_50+Advanced_Augmentation"          # used as prefix in all output filenames
OUT_DIR    = Path(r"C:\Users\Yasmi\OneDrive\Desktop\UNI\This semester\[Projects]\Machine II\[Presentation_Preperation]\Experiments to Present\1.Baseline_50\2.Baseline_50+Advanced_Augmentation")

# ══════════════════════════════════════════════════════════════════════════════
# FIXED — do not change
# ══════════════════════════════════════════════════════════════════════════════

DATA_YAML = Path(r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\DSPCBSD+-1\data.yaml")

# Class labels shown on confusion matrix axes: index.ABBREV
CLASS_LABELS = [
    "0.SH",   # Short
    "1.SP",   # Spur
    "2.SC",   # Spurious Copper
    "3.OP",   # Open
    "4.MB",   # Mouse Bite
    "5.HB",   # Hole Breakout
    "6.CS",   # Conductor Scratch
    "7.CFO",  # Conductor Foreign Object
    "8.BMFO", # Base Material Foreign Object
]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":     150,
})

# 2-row × 5-col layout matching YOLO's results.png style
# (top row = train metrics, bottom row = val metrics)
METRIC_PANELS = [
    # row 0
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    # row 1
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(y, window=5):
    """Simple running average used as the smooth overlay."""
    s = pd.Series(y).rolling(window, min_periods=1, center=True).mean()
    return s.values


def plot_training_curves():
    csv_path = EXP_FOLDER / "results.csv"
    if not csv_path.exists():
        print(f"[SKIP] no results.csv found in {EXP_FOLDER}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    epochs = df["epoch"].values

    # 2 rows × 5 cols — matches YOLO results.png layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(EXP_NAME, fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for i, col in enumerate(METRIC_PANELS):
        ax = axes_flat[i]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        y = df[col].values
        ax.plot(epochs, y,          color="#4472C4", lw=1.2,
                marker="o", markersize=3, label="results")
        ax.plot(epochs, _smooth(y), color="#ED7D31", lw=1.5,
                linestyle="dotted", label="smooth")
        ax.set_title(col)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        if i == 0:                  # only show legend once
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / f"{EXP_NAME}_training_curves.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def _patch_yaml() -> Path:
    """Return a temp data.yaml with class labels (index.ABBREV)."""
    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["names"] = CLASS_LABELS
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=DATA_YAML.parent
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


def _draw_cm(matrix, labels, normalize, title, out_path):
    """Draw confusion matrix with seaborn using our class labels."""
    import seaborn as sns
    import warnings

    # Annotation: blank near-zero cells (matches YOLO style)
    threshold = 0.005 if normalize else 0.5
    fmt_fn = lambda v: (f"{v:.2f}" if normalize else f"{int(v)}") if v > threshold else ""
    annot = np.vectorize(fmt_fn)(matrix)

    fig, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            matrix,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="Blues",
            square=True,
            vmin=0.0,
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            annot_kws={"size": 8},
        ).set_facecolor((1, 1, 1))

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.name}")


CURVE_FILES = [
    "BoxR_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxF1_curve.png",
]


def _run_val(do_cm=True, do_curves=True):
    """
    Run model.val() once and save curve PNGs and/or confusion matrices to OUT_DIR.
    Avoids running validation twice when both --cm and --curves are requested.
    """
    from ultralytics import YOLO

    weights = EXP_FOLDER / "weights" / "best.pt"
    if not weights.exists():
        print(f"[SKIP] no weights/best.pt found in {EXP_FOLDER}")
        return

    captured = {}

    def on_val_end(validator):
        if hasattr(validator, "confusion_matrix") and validator.confusion_matrix is not None:
            captured["matrix"] = validator.confusion_matrix.matrix.copy()

    patched_yaml = _patch_yaml()

    try:
        # Register CoordAtt (and CBAM) into ultralytics.nn.modules.block so
        # pickle can find them when loading checkpoints trained with those modules.
        import sys, importlib
        sys.path.insert(0, str(Path(__file__).parent / "experiments"))
        _pca = importlib.import_module("per_class_analysis")
        _pca._register_custom_modules()

        val_tmp = OUT_DIR / "_val_tmp"
        model = YOLO(str(weights))
        # Overwrite the checkpoint's class names so curve plots use our labels
        name_map = {i: name for i, name in enumerate(CLASS_LABELS)}
        model.model.names = name_map # type: ignore
        model.add_callback("on_val_end", on_val_end)
        model.val(
            data     = str(patched_yaml),
            imgsz    = 640,
            batch    = 6,
            device   = "0",
            amp      = True,
            plots    = True,     # MUST be True — YOLO builds CM + curve plots only when True
            save_dir = str(val_tmp),
            name     = EXP_NAME,
            verbose  = False,
        )

        # ── Curve plots ───────────────────────────────────────────────────────
        if do_curves:
            # When save_dir is explicit, YOLO writes directly there (not save_dir/name)
            yolo_out = val_tmp
            for fname in CURVE_FILES:
                src = yolo_out / fname
                if src.exists():
                    dst = OUT_DIR / f"{EXP_NAME}_{fname}"
                    shutil.copy2(src, dst)
                    print(f"✓ {dst.name}")
                else:
                    print(f"[WARN] {fname} not found in YOLO output — skipping")

        # Done with YOLO's temp dir
        shutil.rmtree(val_tmp, ignore_errors=True)

        # ── Confusion matrices ────────────────────────────────────────────────
        if do_cm:
            if "matrix" not in captured:
                print("[ERROR] confusion matrix not captured — check Ultralytics version")
                return

            cm_raw = captured["matrix"]          # shape (nc+1, nc+1) = (10, 10)
            labels = CLASS_LABELS + ["background"]

            # Raw counts
            _draw_cm(cm_raw, labels, normalize=False,
                     title="Confusion Matrix",
                     out_path=OUT_DIR / f"{EXP_NAME}_confusion_matrix.png")

            # Normalised column-wise (same as YOLO)
            cm_norm = cm_raw.astype(float)
            col_sums = cm_norm.sum(axis=0)
            col_sums[col_sums == 0] = 1
            cm_norm /= col_sums
            _draw_cm(cm_norm, labels, normalize=True,
                     title="Confusion Matrix Normalized",
                     out_path=OUT_DIR / f"{EXP_NAME}_confusion_matrix_normalized.png")

    finally:
        patched_yaml.unlink(missing_ok=True)
        shutil.rmtree(OUT_DIR / "_val_tmp", ignore_errors=True)


def regen_confusion_matrix():
    _run_val(do_cm=True, do_curves=False)


def regen_curve_plots():
    _run_val(do_cm=False, do_curves=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots",  action="store_true", help="Regenerate training curve plots (no GPU)")
    parser.add_argument("--cm",     action="store_true", help="Regenerate confusion matrices (GPU)")
    parser.add_argument("--curves", action="store_true", help="Regenerate BoxR/P/PR/F1 curve plots (GPU)")
    args = parser.parse_args()

    if not args.plots and not args.cm and not args.curves:
        parser.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment : {EXP_NAME}")
    print(f"Folder     : {EXP_FOLDER}")
    print(f"Output     : {OUT_DIR}\n")

    if args.plots:
        plot_training_curves()

    # --cm and --curves share one val run if both are requested
    if args.cm or args.curves:
        _run_val(do_cm=args.cm, do_curves=args.curves)

    print("\nDone.")


if __name__ == "__main__":
    main()
