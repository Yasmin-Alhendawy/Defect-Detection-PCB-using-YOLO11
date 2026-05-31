"""
Oversample images containing classes 2, 6, 7 for DSPCBSD+.
Creates an oversampled copy of the train/ folder so the original stays clean.

Output: DSPCBSD+-1/train_os/{images,labels}  + data_os.yaml pointing at it.

Strategy:
- Count instances per class in train.
- For each image, look at which target classes appear.
- Duplicate the image+label by a factor based on the rarest target class it contains.
- Target boost: try to bring 2/6/7 instance counts to ~80% of the majority class.
"""

from pathlib import Path
import shutil
from collections import Counter

# --- config -----------------------------------------------------------------
ROOT = Path(r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\DSPCBSD+-1")
SRC_IMG = ROOT / "train" / "images"
SRC_LBL = ROOT / "train" / "labels"
DST_IMG = ROOT / "train_os" / "images"
DST_LBL = ROOT / "train_os" / "labels"
TARGET_CLASSES = {2, 6, 7}
TARGET_RATIO = 0.80          # bring rare classes up to 80% of majority count
MAX_DUP = 8                  # safety cap so a single image isn't copied 50x
# ----------------------------------------------------------------------------

DST_IMG.mkdir(parents=True, exist_ok=True)
DST_LBL.mkdir(parents=True, exist_ok=True)

# 1. count instances per class
counts = Counter()
img_to_classes = {}
for lbl in SRC_LBL.glob("*.txt"):
    classes_in_img = []
    with lbl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            counts[cid] += 1
            classes_in_img.append(cid)
    img_to_classes[lbl.stem] = classes_in_img

print("Original class instance counts:")
for c in sorted(counts):
    print(f"  class {c}: {counts[c]}")

majority = max(counts.values())
target_count = int(majority * TARGET_RATIO)
print(f"\nMajority class count: {majority}")
print(f"Target count for classes {TARGET_CLASSES}: {target_count}")

# 2. compute boost factor per target class
boost = {}
for c in TARGET_CLASSES:
    if counts[c] >= target_count:
        boost[c] = 1
    else:
        boost[c] = min(MAX_DUP, max(1, round(target_count / max(1, counts[c]))))
print(f"\nDuplication factors: {boost}")

# 3. for each image, duplication = max boost across target classes it contains
n_copied = 0
n_duplicated_extra = 0
for stem, classes_in_img in img_to_classes.items():
    targets_present = TARGET_CLASSES & set(classes_in_img)
    factor = max((boost[c] for c in targets_present), default=1)

    # find source image (any extension)
    src_imgs = list(SRC_IMG.glob(stem + ".*"))
    if not src_imgs:
        continue
    src_img = src_imgs[0]
    src_lbl = SRC_LBL / (stem + ".txt")

    # original copy
    shutil.copy2(src_img, DST_IMG / src_img.name)
    shutil.copy2(src_lbl, DST_LBL / src_lbl.name)
    n_copied += 1

    # duplicates with suffix
    for i in range(1, factor):
        new_stem = f"{stem}_dup{i}"
        shutil.copy2(src_img, DST_IMG / (new_stem + src_img.suffix))
        shutil.copy2(src_lbl, DST_LBL / (new_stem + ".txt"))
        n_duplicated_extra += 1

print(f"\nCopied {n_copied} originals + {n_duplicated_extra} duplicates "
      f"= {n_copied + n_duplicated_extra} total train images")

# 4. recount after oversampling
new_counts = Counter()
for lbl in DST_LBL.glob("*.txt"):
    with lbl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                new_counts[int(line.split()[0])] += 1
print("\nPost-oversample class instance counts:")
for c in sorted(new_counts):
    delta = new_counts[c] - counts[c]
    print(f"  class {c}: {new_counts[c]} (+{delta})")

# 5. write data_os.yaml
data_os = ROOT / "data_os.yaml"
data_os.write_text(
    "train: ../train_os/images\n"
    "val: ../valid/images\n"
    "test: ../test/images\n\n"
    "nc: 9\n"
    "names: ['0','1','2','3','4','5','6','7','8']\n"
)
print(f"\nWrote {data_os}")
print("Train with: data=data_os.yaml")
