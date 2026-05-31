"""
compare_experiments.py — IEEE-paper-ready comparison plots across all PCB experiments.

Outputs (PNG, 300 dpi) in runs/detect/pcb_runs/_comparison/:
  1. training_curves.png       — mAP50 and mAP50-95 over epochs, all experiments
  2. loss_curves.png           — box/cls/dfl train losses over epochs
  3. final_metrics_bar.png     — final P / R / mAP50 / mAP50-95 per experiment
  4. per_class_map_bar.png     — per-class mAP50 per experiment (from validation)
  5. confusion_matrices_grid.png — side-by-side normalized confusion matrices
  6. summary_table.csv         — numeric summary for the paper
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ultralytics import YOLO

# --- config -----------------------------------------------------------------
ROOT = Path(r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD")
RUNS = ROOT / "runs" / "detect" / "pcb_runs"
DATA = ROOT / "DSPCBSD+-1" / "data.yaml"
OUT  = RUNS / "_comparison"
OUT.mkdir(exist_ok=True)

# Experiments to compare — edit this list as you add runs
EXPERIMENTS = [
    "expA_fixed_baseline",
    "expB_cbam",
    "expB_cbam_pretrained",
    "expB_cbam_efficient",
    "expB_cbam_stackedA",
    "exp2a_oversample_267",   # comment out if not finished yet
]

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8']
# ----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# keep only experiments that exist
exps = [e for e in EXPERIMENTS if (RUNS / e / "results.csv").exists()]
print(f"Comparing {len(exps)} experiments: {exps}\n")

# load results.csv for each
dfs = {e: pd.read_csv(RUNS / e / "results.csv") for e in exps}
for e, df in dfs.items():
    df.columns = [c.strip() for c in df.columns]

colors = plt.cm.tab10(np.linspace(0, 1, len(exps)))

# === 1. Training curves (mAP) =================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for (e, df), c in zip(dfs.items(), colors):
    axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], label=e, color=c, lw=1.5)
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label=e, color=c, lw=1.5)
axes[0].set(xlabel="Epoch", ylabel="mAP@0.5", title="Validation mAP@0.5 over Epochs")
axes[1].set(xlabel="Epoch", ylabel="mAP@0.5:0.95", title="Validation mAP@0.5:0.95 over Epochs")
for ax in axes:
    ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "training_curves.png", bbox_inches="tight")
plt.close(fig)
print("[1/6] training_curves.png")

# === 2. Loss curves ===========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for (e, df), c in zip(dfs.items(), colors):
    axes[0].plot(df["epoch"], df["train/box_loss"], label=e, color=c, lw=1.2)
    axes[1].plot(df["epoch"], df["train/cls_loss"], label=e, color=c, lw=1.2)
    axes[2].plot(df["epoch"], df["train/dfl_loss"], label=e, color=c, lw=1.2)
axes[0].set(xlabel="Epoch", ylabel="Box Loss", title="Train Box Loss")
axes[1].set(xlabel="Epoch", ylabel="Cls Loss", title="Train Cls Loss")
axes[2].set(xlabel="Epoch", ylabel="DFL Loss", title="Train DFL Loss")
axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "loss_curves.png", bbox_inches="tight")
plt.close(fig)
print("[2/6] loss_curves.png")

# === 3. Final metrics bar chart ===============================================
final = []
for e, df in dfs.items():
    last = df.iloc[-1]
    final.append({
        "experiment": e,
        "precision": last["metrics/precision(B)"],
        "recall":    last["metrics/recall(B)"],
        "mAP50":     last["metrics/mAP50(B)"],
        "mAP50-95":  last["metrics/mAP50-95(B)"],
    })
fdf = pd.DataFrame(final)

fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(exps)), 5))
x = np.arange(len(exps))
w = 0.2
ax.bar(x - 1.5*w, fdf["precision"], w, label="Precision")
ax.bar(x - 0.5*w, fdf["recall"],    w, label="Recall")
ax.bar(x + 0.5*w, fdf["mAP50"],     w, label="mAP@0.5")
ax.bar(x + 1.5*w, fdf["mAP50-95"],  w, label="mAP@0.5:0.95")
ax.set_xticks(x)
ax.set_xticklabels(fdf["experiment"], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Final Validation Metrics by Experiment")
ax.legend()
ax.set_ylim(0, 1.0)
for i, row in fdf.iterrows():
    ax.text(i + 1.5*w, row["mAP50-95"] + 0.01, f'{row["mAP50-95"]:.3f}',
            ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "final_metrics_bar.png", bbox_inches="tight")
plt.close(fig)
print("[3/6] final_metrics_bar.png")

# === 4. Per-class mAP (requires running validation) ==========================
print("\n[4/6] Running validation for per-class mAP (this may take a few min)...")
per_class = {}
for e in exps:
    w = RUNS / e / "weights" / "best.pt"
    if not w.exists():
        print(f"  skip {e}: no best.pt")
        continue
    print(f"  validating {e}...")
    model = YOLO(str(w))
    res = model.val(data=str(DATA), imgsz=640, batch=8, plots=False, verbose=False,
                    project=str(OUT / "_tmp_val"), name=e, exist_ok=True)
    # res.box.maps is per-class mAP50-95 array
    per_class[e] = res.box.maps  # length = nc

if per_class:
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(CLASS_NAMES) * len(per_class) / 4), 5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(per_class)
    for i, (e, vals) in enumerate(per_class.items()):
        offs = (i - len(per_class) / 2) * w + w / 2
        ax.bar(x + offs, vals, w, label=e, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Class")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Per-Class mAP@0.5:0.95 by Experiment")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_map_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("[4/6] per_class_map_bar.png")

    # add per-class to summary
    for e, vals in per_class.items():
        for i, c in enumerate(CLASS_NAMES):
            fdf.loc[fdf["experiment"] == e, f"mAP50-95_class_{c}"] = vals[i]

# === 5. Confusion matrices grid ==============================================
cms = [(e, RUNS / e / "confusion_matrix_normalized.png") for e in exps
       if (RUNS / e / "confusion_matrix_normalized.png").exists()]
if cms:
    n = len(cms)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (e, p) in zip(axes, cms):
        ax.imshow(mpimg.imread(p))
        ax.set_title(e, fontsize=9)
        ax.axis("off")
    for ax in axes[len(cms):]:
        ax.axis("off")
    fig.suptitle("Normalized Confusion Matrices", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices_grid.png", bbox_inches="tight")
    plt.close(fig)
    print("[5/6] confusion_matrices_grid.png")

# === 6. Summary CSV ==========================================================
fdf.to_csv(OUT / "summary_table.csv", index=False)
print(f"[6/6] summary_table.csv\n")
print(fdf[["experiment", "precision", "recall", "mAP50", "mAP50-95"]].to_string(index=False))
print(f"\nAll outputs in: {OUT}")
