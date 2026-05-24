# =============================================================================
# Experiment 1 — Baseline + Advanced Augmentation
# Model: YOLOv11s (yolo11s.pt)
# Dataset: Deep Surface PCB Defect Dataset (DSPCBSD)
# Purpose: Ablation study — isolating data-level interventions only.
#          Architecture is identical to Exp. 0 (train.py).
#
# Three data-level changes vs Exp. 0:
#   [A] Custom Albumentations pipeline injected into Ultralytics dataset loader
#   [B] Minority class copy-paste augmentation (preprocessing step)
#   [C] Class imbalance penalty via cls_pw (fl_gamma not supported in 8.4.52)
# =============================================================================

import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

import albumentations as A
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

# =============================================================================
# [A] CUSTOM ALBUMENTATIONS PIPELINE
# Subclass YOLODataset and override build_transforms() to inject our pipeline.
# bbox_params ensures bounding boxes are updated after every spatial transform.
# =============================================================================

class PCBDatasetWithAugmentation(YOLODataset):
    """Custom dataset that injects Albumentations for Experiment 1."""

    def build_transforms(self, hyp=None):
        albu_transform = A.Compose(
            [
                # [A1] Random 90-degree rotation — handles PCB orientation variance
                A.RandomRotate90(p=0.5),
                # [A2] CLAHE — enhances local contrast to highlight subtle defects
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                # [A3] ElasticTransform — simulates physical PCB surface deformation
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                # [A4] CoarseDropout — forces model to detect from partial observations
                # Use a conservative CoarseDropout signature to avoid compatibility issues
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,
            ),
        )

        parent_transforms = super().build_transforms(hyp)

        class AlbuWrapper:
            def __init__(self, albu, parent):
                self.albu = albu
                self.parent = parent

            def __call__(self, labels):
                img = labels["img"]
                bboxes = labels.get("bboxes", np.zeros((0, 4)))
                classes = labels.get("cls", np.zeros((0,)))
                bbox_list = bboxes.tolist() if len(bboxes) else []
                class_list = classes.tolist() if len(classes) else []
                try:
                    transformed = self.albu(
                        image=img,
                        bboxes=bbox_list,
                        class_labels=class_list,
                    )
                    labels["img"] = transformed["image"]
                    if len(transformed["bboxes"]) > 0:
                        labels["bboxes"] = np.array(transformed["bboxes"], dtype=np.float32)
                        labels["cls"] = np.array(transformed["class_labels"], dtype=np.float32)
                except Exception as e:
                    LOGGER.warning(f"Albumentations transform failed: {e}")
                return self.parent(labels)

        return AlbuWrapper(albu_transform, parent_transforms)


# =============================================================================
# [B] MINORITY CLASS COPY-PASTE AUGMENTATION
# Runs once before training. Crops minority defect instances and pastes
# them onto background images to generate synthetic training samples.
# =============================================================================

def run_copy_paste_augmentation(
    images_dir, labels_dir,
    num_synthetic=200,
    min_class_threshold_ratio=0.5,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    LOGGER.info("[B] Starting minority class copy-paste augmentation...")

    class_instances = defaultdict(list)
    for label_file in labels_dir.glob("*.txt"):
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = images_dir / (label_file.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        for line in label_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            class_instances[cls_id].append((img_path, cx, cy, w, h))

    if not class_instances:
        LOGGER.warning("[B] No annotations found. Skipping copy-paste.")
        return 0

    class_counts = {cls: len(v) for cls, v in class_instances.items()}
    mean_count = np.mean(list(class_counts.values()))
    threshold = mean_count * min_class_threshold_ratio
    minority_classes = [c for c, n in class_counts.items() if n < threshold]

    LOGGER.info(f"[B] Class counts: {class_counts}")
    LOGGER.info(f"[B] Mean: {mean_count:.1f} | Threshold: {threshold:.1f}")
    LOGGER.info(f"[B] Minority classes: {minority_classes}")

    if not minority_classes:
        LOGGER.info("[B] No minority classes found. Skipping copy-paste.")
        return 0

    label_files = list(labels_dir.glob("*.txt"))
    background_images = [
        images_dir / (lf.stem + ext)
        for lf in label_files
        for ext in [".jpg", ".jpeg", ".png"]
        if (images_dir / (lf.stem + ext)).exists()
    ]

    all_minority = []
    for cls in minority_classes:
        for (img_path, cx, cy, w, h) in class_instances[cls]:
            all_minority.append((cls, img_path, cx, cy, w, h))

    synthetic_count = 0
    for i in range(num_synthetic):
        bg_img_path = random.choice(background_images)
        bg_label_path = labels_dir / (bg_img_path.stem + ".txt")
        bg_img = cv2.imread(str(bg_img_path))
        if bg_img is None:
            continue
        bg_h, bg_w = bg_img.shape[:2]

        cls_id, src_img_path, cx, cy, w, h = random.choice(all_minority)
        src_img = cv2.imread(str(src_img_path))
        if src_img is None:
            continue
        src_h, src_w = src_img.shape[:2]

        x1 = max(0, int(cx * src_w - w * src_w / 2))
        y1 = max(0, int(cy * src_h - h * src_h / 2))
        x2 = min(src_w, x1 + int(w * src_w))
        y2 = min(src_h, y1 + int(h * src_h))
        crop = src_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        scale = min(bg_w / 4 / max(crop.shape[1], 1), bg_h / 4 / max(crop.shape[0], 1), 1.0)
        new_w = max(1, int(crop.shape[1] * scale))
        new_h = max(1, int(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h))

        px = random.randint(0, max(0, bg_w - new_w))
        py = random.randint(0, max(0, bg_h - new_h))
        synthetic_img = bg_img.copy()
        synthetic_img[py:py + new_h, px:px + new_w] = crop

        new_cx = (px + new_w / 2) / bg_w
        new_cy = (py + new_h / 2) / bg_h
        new_wn = new_w / bg_w
        new_hn = new_h / bg_h

        syn_name = f"synthetic_cp_{i:04d}"
        cv2.imwrite(str(images_dir / (syn_name + ".jpg")), synthetic_img)

        existing = bg_label_path.read_text().strip().splitlines() if bg_label_path.exists() else []
        with open(labels_dir / (syn_name + ".txt"), "w") as f:
            f.write("\n".join(existing + [f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_wn:.6f} {new_hn:.6f}"]))

        synthetic_count += 1

    LOGGER.info(f"[B] Done. {synthetic_count} synthetic images created.")
    return synthetic_count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # [B] Copy-paste augmentation — runs before training
    print("=" * 60)
    print("Experiment 1 — Step 1: Copy-Paste Augmentation")
    print("=" * 60)
    n = run_copy_paste_augmentation(
        images_dir="DSPCBSD+-1/train/images",
        labels_dir="DSPCBSD+-1/train/labels",
        num_synthetic=200,
        seed=42,
    )
    print(f"[B] {n} synthetic images added.\n")

    # [A] Monkey-patch DetectionTrainer to use our custom dataset class
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    original_build_dataset = DetectionTrainer.build_dataset

    def custom_build_dataset(self, img_path, mode="train", batch=None):
        """[A] Use PCBDatasetWithAugmentation for train split only."""
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        if mode == "train":
            return PCBDatasetWithAugmentation(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=False,
                cache=self.args.cache or False,
                single_cls=self.args.single_cls or False,
                stride=int(gs),
                pad=0.0,
                prefix="train: ",
                task="detect",          # fixed: hardcoded instead of self.task
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction,
            )
        else:
            return original_build_dataset(self, img_path, mode, batch)

    DetectionTrainer.build_dataset = custom_build_dataset

    # [A] + [C] Train
    print("=" * 60)
    print("Experiment 1 — Step 2: Training")
    print("=" * 60)

    model = YOLO("yolo11s.pt")
    print("Model loaded: yolo11s.pt")

    results = model.train(
        data="DSPCBSD+-1/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        workers=4,
        project="pcb_runs",              # fixed: no double path
        name="exp1_augmentation",
        # [C] cls_pw upweights classification loss for imbalanced classes
        # replaces fl_gamma which is unsupported in Ultralytics 8.4.52
        cls_pw=2.0,
        erasing=0.0,                     # disable default erasing (we use CoarseDropout)
    )

    print("\n" + "=" * 60)
    print("Experiment 1 Complete!")
    print("Results saved to: pcb_runs/exp1_augmentation")
    print("=" * 60)