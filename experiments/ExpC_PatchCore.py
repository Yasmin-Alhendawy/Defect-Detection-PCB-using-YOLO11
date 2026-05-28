"""
Exp C — PatchCore (ResNet-18) Unsupervised Baseline
====================================================
Framing for the paper:
  - Unsupervised method: no bounding-box labels used during training
  - Builds a patch-level feature memory bank from ALL training images
  - At test time: scores each patch against the memory bank (kNN distance)
  - Produces a spatial anomaly heatmap → thresholded → bounding-box predictions
  - Compared against YOLOv11+CBAM on the same test split

Outputs (saved to runs/detect/pcb_runs/expC_patchcore/):
  - patchcore_results.csv   — per-image anomaly scores + image-level metrics
  - patchcore_summary.csv   — overall metrics for the paper table
  - sample_heatmaps/        — visualisation for a few test images

Requirements (install once):
  pip install torch torchvision scikit-learn tqdm Pillow opencv-python
"""

import os, csv, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent          # DSPCBSD root
DATA       = BASE / "DSPCBSD+-1"
TRAIN_IMG  = DATA / "train" / "images"
TRAIN_LBL  = DATA / "train" / "labels"
TEST_IMG   = DATA / "test"  / "images"
TEST_LBL   = DATA / "test"  / "labels"
OUT_DIR    = BASE / "runs" / "detect" / "pcb_runs" / "expC_patchcore"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR = OUT_DIR / "sample_heatmaps"
HEATMAP_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE        = 224          # standard ImageNet size
BATCH_SIZE      = 32           # safe for 4GB VRAM
CORESET_RATIO   = 0.05         # keep 5% of patches in memory bank (~balance speed/accuracy)
KNN_K           = 5            # nearest neighbours for anomaly score
SCORE_THRESHOLD = 0.5          # normalised score threshold for box prediction
IOU_THRESHOLD   = 0.1          # IoU for a "hit" (loose — anomaly maps are rough)
MAX_HEATMAPS    = 20           # save this many visualisation images
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ─── Image transform ──────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])

# ─── Feature extractor (ResNet-18, layer2 + layer3) ───────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # stem + layer1 + layer2
        self.early = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2          # output: B×128×28×28
        )
        self.layer3 = backbone.layer3                 # output: B×256×14×14

    def forward(self, x):
        f2 = self.early(x)                            # 28×28×128
        f3 = self.layer3(f2)                          # 14×14×256
        # upsample f3 to match f2 spatial size
        f3_up = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                          align_corners=False)
        return torch.cat([f2, f3_up], dim=1)          # 28×28×384

extractor = FeatureExtractor().to(DEVICE).eval()


def extract_features(img_paths, desc="Extracting"):
    """Returns tensor (N_patches_total, 384) and list of per-image patch counts."""
    all_feats, counts = [], []
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc=desc):
        batch_paths = img_paths[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception:
                imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            feat = extractor(batch)                   # B×384×28×28
        B, C, H, W = feat.shape
        # reshape to (B, H*W, C) — each spatial location = one patch
        patches = feat.permute(0, 2, 3, 1).reshape(B, H*W, C).cpu()
        for j in range(B):
            all_feats.append(patches[j])              # (784, 384)
            counts.append(H * W)
    return torch.cat(all_feats, dim=0), counts        # (N_total_patches, 384)


# ─── Step 1: Build memory bank from training images ───────────────────────────
print("\n=== Step 1: Building memory bank ===")
train_imgs = sorted([p for p in TRAIN_IMG.glob("*.jpg")])
print(f"Training images: {len(train_imgs)}")

t0 = time.time()
train_feats, train_counts = extract_features(train_imgs, "Train features")
print(f"  Total train patches: {train_feats.shape[0]:,}  ({time.time()-t0:.0f}s)")

# Coreset subsampling — random for speed (greedy coreset is slower but marginally better)
n_keep = max(1000, int(len(train_feats) * CORESET_RATIO))
idx    = torch.randperm(len(train_feats))[:n_keep]
memory_bank = train_feats[idx]                        # (n_keep, 384)
print(f"  Memory bank size: {memory_bank.shape[0]:,} patches (ratio={CORESET_RATIO})")

# L2-normalise for cosine-like distance
memory_bank_norm = nn.functional.normalize(memory_bank, dim=1).to(DEVICE)


# ─── Step 2: Score test images ────────────────────────────────────────────────
print("\n=== Step 2: Scoring test images ===")
test_imgs = sorted([p for p in TEST_IMG.glob("*.jpg")])
print(f"Test images: {len(test_imgs)}")

def score_patches(patches_tensor):
    """
    patches_tensor: (H*W, 384) for one image
    Returns anomaly score per patch (H*W,) — higher = more anomalous.
    """
    patches_norm = nn.functional.normalize(patches_tensor.to(DEVICE), dim=1)
    # Cosine distance = 1 - cosine similarity
    # matrix: (H*W, n_keep)
    sim = torch.mm(patches_norm, memory_bank_norm.T)   # (784, n_keep)
    cos_dist = 1.0 - sim                               # (784, n_keep)
    # kNN: mean of k smallest distances
    topk = torch.topk(cos_dist, k=KNN_K, dim=1, largest=False).values
    return topk.mean(dim=1).cpu()                      # (784,)


def load_gt_boxes(label_path, img_w, img_h):
    """Returns list of (x1,y1,x2,y2) in pixel coords from YOLO label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw/2) * img_w
            y1 = (cy - bh/2) * img_h
            x2 = (cx + bw/2) * img_w
            y2 = (cy + bh/2) * img_h
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


def heatmap_to_boxes(score_map_norm, img_w, img_h, threshold):
    """
    score_map_norm: (H, W) float in [0,1]
    Returns list of (x1,y1,x2,y2) pixel boxes from connected components.
    """
    binary = (score_map_norm >= threshold).astype(np.uint8) * 255
    # small morphological close to join nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for i in range(1, n_labels):   # skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < 50:              # skip tiny noise blobs
            continue
        # scale to original image size
        sx = img_w / score_map_norm.shape[1]
        sy = img_h / score_map_norm.shape[0]
        boxes.append((x*sx, y*sy, (x+w)*sx, (y+h)*sy))
    return boxes


# Process each test image
PATCH_H = PATCH_W = 28          # feature map spatial size for 224×224 input
results = []
n_saved = 0

all_image_scores = []           # image-level anomaly scores (for AUROC proxy)
TP_total = FP_total = FN_total = 0

for img_path in tqdm(test_imgs, desc="Scoring test"):
    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    img_w, img_h = img_pil.size

    # Extract features
    inp = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = extractor(inp)                             # 1×384×28×28
    patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, 384).cpu()

    # Score
    scores = score_patches(patches)                       # (784,)
    score_map = scores.reshape(PATCH_H, PATCH_W).numpy()

    # Normalise score map to [0, 1] using global 5th–95th percentile
    # (normalise per-image so threshold is comparable)
    pmin, pmax = np.percentile(score_map, 5), np.percentile(score_map, 95)
    if pmax > pmin:
        score_map_norm = np.clip((score_map - pmin) / (pmax - pmin), 0, 1)
    else:
        score_map_norm = np.zeros_like(score_map)

    image_score = float(scores.max())                    # image-level: max patch score
    all_image_scores.append(image_score)

    # GT boxes
    stem = img_path.stem
    lbl_path = TEST_LBL / (stem + ".txt")
    gt_boxes = load_gt_boxes(lbl_path, img_w, img_h)

    # Predicted boxes
    pred_boxes = heatmap_to_boxes(score_map_norm, img_w, img_h, SCORE_THRESHOLD)

    # Evaluate: for each GT box, check if any pred box hits it (IoU > threshold)
    tp = fp = fn = 0
    matched_gt = set()
    for pb in pred_boxes:
        hit = False
        for gi, gb in enumerate(gt_boxes):
            if gi not in matched_gt and iou(pb, gb) >= IOU_THRESHOLD:
                matched_gt.add(gi)
                hit = True
                break
        if hit:
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    TP_total += tp; FP_total += fp; FN_total += fn

    results.append({
        "image":        img_path.name,
        "image_score":  round(image_score, 5),
        "n_gt_boxes":   len(gt_boxes),
        "n_pred_boxes": len(pred_boxes),
        "TP": tp, "FP": fp, "FN": fn,
    })

    # Save heatmap visualisation
    if n_saved < MAX_HEATMAPS:
        img_np = np.array(img_pil.resize((img_w, img_h)))
        hmap_resized = cv2.resize(score_map_norm, (img_w, img_h))
        hmap_color = cv2.applyColorMap((hmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.5, hmap_color, 0.5, 0)
        # draw GT boxes (green) and pred boxes (red)
        for (x1,y1,x2,y2) in gt_boxes:
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        for (x1,y1,x2,y2) in pred_boxes:
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        out_path = HEATMAP_DIR / f"{stem}_heatmap.jpg"
        cv2.imwrite(str(out_path), overlay)
        n_saved += 1


# ─── Step 3: Compute metrics ──────────────────────────────────────────────────
print("\n=== Step 3: Results ===")

precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
recall    = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Image-level detection rate (all test images ARE defective, so this is recall@image)
# A "detected" image has image_score above median → trivially all detected
# More useful: what fraction of images have at least 1 TP box
imgs_with_tp = sum(1 for r in results if r["TP"] > 0)
image_detection_rate = imgs_with_tp / len(results) if results else 0

print(f"  Patch-level box detection (IoU≥{IOU_THRESHOLD}):")
print(f"    Precision : {precision:.4f}")
print(f"    Recall    : {recall:.4f}")
print(f"    F1        : {f1:.4f}")
print(f"  Image-level detection rate: {image_detection_rate:.4f}")
print(f"  TP={TP_total}  FP={FP_total}  FN={FN_total}")

# Save per-image results
with open(OUT_DIR / "patchcore_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Save summary
summary = {
    "method":               "PatchCore-ResNet18",
    "memory_bank_patches":  int(memory_bank.shape[0]),
    "coreset_ratio":        CORESET_RATIO,
    "knn_k":                KNN_K,
    "score_threshold":      SCORE_THRESHOLD,
    "iou_threshold":        IOU_THRESHOLD,
    "n_test_images":        len(test_imgs),
    "precision":            round(precision, 4),
    "recall":               round(recall, 4),
    "f1":                   round(f1, 4),
    "image_detection_rate": round(image_detection_rate, 4),
    "TP_total":             TP_total,
    "FP_total":             FP_total,
    "FN_total":             FN_total,
}
with open(OUT_DIR / "patchcore_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary.keys())
    writer.writeheader()
    writer.writerow(summary)

print(f"\nResults saved to: {OUT_DIR}")
print(f"Heatmaps saved to: {HEATMAP_DIR}  ({n_saved} images)")
print("\n=== Done ===")
