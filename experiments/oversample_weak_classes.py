"""
Oversample weak classes (6 & 7) in the training set.
Safe: never deletes originals. Creates copies with prefix 'os2_' (2x) or 'os3_' (3x).
Run BEFORE training Exp B.
"""

import os
import shutil
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path(__file__).parent.parent / "DSPCBSD+-1"
TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATASET_ROOT / "train" / "labels"

TARGET_CLASSES = {6, 7}   # classes to oversample
COPIES         = 2         # 2 = duplicate (total 2x); 3 = triplicate (total 3x)
DRY_RUN        = False     # set True to preview without writing files
# ──────────────────────────────────────────────────────────────────────────────


def get_classes(lbl_path: Path) -> set:
    classes = set()
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def main():
    label_files = sorted(TRAIN_LBL_DIR.glob("*.txt"))
    print(f"Total label files: {len(label_files)}")

    # Skip files that are already copies we made
    prefixes = tuple(f"os{i}_" for i in range(2, COPIES + 1))
    originals = [f for f in label_files if not f.stem.startswith(prefixes)]

    targets = [f for f in originals if TARGET_CLASSES & get_classes(f)]
    print(f"Images containing class {TARGET_CLASSES}: {len(targets)}")

    if DRY_RUN:
        print("\n[DRY RUN] Would create:")

    added_imgs, added_lbls, skipped = 0, 0, 0

    for copy_n in range(2, COPIES + 1):
        prefix = f"os{copy_n}_"
        for lbl in targets:
            # Find the image (jpg only — skip .npy)
            img = TRAIN_IMG_DIR / (lbl.stem + ".jpg")
            if not img.exists():
                skipped += 1
                continue

            new_img = TRAIN_IMG_DIR / (prefix + img.name)
            new_lbl = TRAIN_LBL_DIR / (prefix + lbl.name)

            if DRY_RUN:
                print(f"  {new_img.name}")
                continue

            shutil.copy2(img, new_img)
            shutil.copy2(lbl, new_lbl)
            added_imgs += 1
            added_lbls += 1

    if not DRY_RUN:
        print(f"\nDone. Added {added_imgs} images + {added_lbls} labels.")
        if skipped:
            print(f"Skipped {skipped} labels (image not found).")

        # Report new class distribution
        all_labels = list(TRAIN_LBL_DIR.glob("*.txt"))
        counts = Counter()
        for f in all_labels:
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        total = sum(counts.values())
        print(f"\nNew training distribution ({total} total instances):")
        for cls in sorted(counts):
            print(f"  Class {cls}: {counts[cls]:5d}  ({counts[cls]/total*100:.1f}%)")


if __name__ == "__main__":
    main()
