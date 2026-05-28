# Quick path checker — run before training to confirm everything exists.
from pathlib import Path

print("=" * 55)
print("PATH CHECK FOR EXPERIMENT 1")
print("=" * 55)

checks = {
    "Dataset config (data.yaml)": "DSPCBSD+-1/data.yaml",
    "Train images folder":        "DSPCBSD+-1/train/images",
    "Train labels folder":        "DSPCBSD+-1/train/labels",
    "Valid images folder":        "DSPCBSD+-1/valid/images",
    "Valid labels folder":        "DSPCBSD+-1/valid/labels",
    "Pretrained weights":         "yolo11s.pt",
    "Training script":            "experiments/exp1_augmentation/train_exp1.py",
}

all_good = True
for name, path in checks.items():
    p = Path(path)
    if p.exists():
        print(f"  OK    {name}")
    else:
        print(f"  MISSING  {name}  ->  {path}")
        all_good = False

# Count files in train folders
img_dir = Path("DSPCBSD+-1/train/images")
lbl_dir = Path("DSPCBSD+-1/train/labels")
if img_dir.exists():
    n_img = len(list(img_dir.glob("*.*")))
    print(f"\n  Train images count: {n_img}")
if lbl_dir.exists():
    n_lbl = len(list(lbl_dir.glob("*.txt")))
    print(f"  Train labels count: {n_lbl}")

print("=" * 55)
if all_good:
    print("ALL PATHS OK — safe to run training.")
else:
    print("SOME PATHS MISSING — fix before running.")
print("=" * 55)