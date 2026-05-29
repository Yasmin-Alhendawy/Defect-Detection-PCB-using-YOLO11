"""
Exp D — SAHI (Slicing Aided Hyper Inference)
=============================================
SAHI slices each test image into overlapping tiles, runs YOLO on each tile
separately, then merges detections back. Small defects that are nearly
invisible at full 640×640 resolution become clearly visible inside a tile.

This is inference-only — zero retraining. Runs on any trained weights.

Usage:
    # On Exp A (default)
    python experiments/ExpD_SAHI.py

    # On fixed Exp B
    python experiments/ExpD_SAHI.py --weights runs/detect/pcb_runs/expB_cbam_pretrained/weights/best.pt

    # With tuned per-class thresholds
    python experiments/ExpD_SAHI.py --thresholds pcb_runs/expA_fixed_baseline/optimal_thresholds.json

Output (saved alongside the weights):
    sahi_results.csv      — per-image TP/FP/FN
    sahi_summary.csv      — overall P/R/F1 + mAP for paper table
    sahi_vs_baseline.csv  — direct comparison: standard vs SAHI

Requirements:
    pip install sahi
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=
    "runs/detect/pcb_runs/expA_fixed_baseline/weights/best.pt")
parser.add_argument("--thresholds", default=None,
    help="Path to optimal_thresholds.json (from ExpF). If omitted, uses conf=0.25.")
parser.add_argument("--slice_size",  type=int, default=320)
parser.add_argument("--overlap",     type=float, default=0.2)
parser.add_argument("--conf",        type=float, default=0.25,
    help="Global conf threshold (used if --thresholds not provided)")
parser.add_argument("--iou_nms",     type=float, default=0.45)
parser.add_argument("--iou_match",   type=float, default=0.5,
    help="IoU threshold to count a detection as TP")
parser.add_argument("--device",      default="0")
parser.add_argument("--imgsz",       type=int, default=640)
args = parser.parse_args()

WEIGHTS    = Path(args.weights)
OUT_DIR    = WEIGHTS.parent.parent   # run folder
NC         = 9
IOU_MATCH  = args.iou_match

# ── Load per-class thresholds (or use global) ──────────────────────────────
if args.thresholds and Path(args.thresholds).exists():
    with open(args.thresholds) as f:
        raw = json.load(f)
    per_class_thresh = {int(k): float(v) for k, v in raw.items()}
    default_conf = min(per_class_thresh.values()) * 0.8  # low enough to capture all
    print(f"Using per-class thresholds from {args.thresholds}")
    print(f"  {per_class_thresh}")
else:
    per_class_thresh = {c: args.conf for c in range(NC)}
    default_conf = args.conf
    print(f"Using global conf={args.conf} for all classes")

# ── Install / import SAHI ──────────────────────────────────────────────────
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    import subprocess, sys
    print("Installing SAHI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sahi", "-q"])
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

# ── Load SAHI model ────────────────────────────────────────────────────────
print(f"\nLoading model: {WEIGHTS}")
detection_model = AutoDetectionModel.from_pretrained(
    model_type      = "ultralytics",
    model_path      = str(WEIGHTS),
    confidence_threshold = default_conf,   # SAHI pre-filters at this level
    device          = f"cuda:{args.device}" if args.device.isdigit() else args.device,
)

# ── Dataset paths ──────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
TEST_IMG  = BASE / "DSPCBSD+-1" / "test" / "images"
TEST_LBL  = BASE / "DSPCBSD+-1" / "test" / "labels"
test_imgs = sorted(TEST_IMG.glob("*.jpg"))
print(f"Test images: {len(test_imgs)}")

# ── Helpers ────────────────────────────────────────────────────────────────
def load_gt_boxes(stem, img_w, img_h):
    lbl = TEST_LBL / (stem + ".txt")
    boxes = []
    if lbl.exists():
        with open(lbl) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5: continue
                cls = int(p[0])
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append((cls,
                               (cx-bw/2)*img_w, (cy-bh/2)*img_h,
                               (cx+bw/2)*img_w, (cy+bh/2)*img_h))
    return boxes

def box_iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0.0

def compute_metrics(results_list):
    TP=FP=FN=0
    for r in results_list:
        TP+=r["TP"]; FP+=r["FP"]; FN+=r["FN"]
    P = TP/(TP+FP) if (TP+FP)>0 else 0
    R = TP/(TP+FN) if (TP+FN)>0 else 0
    F1= 2*P*R/(P+R) if (P+R)>0 else 0
    return P, R, F1

# ── Run SAHI on test set ───────────────────────────────────────────────────
print(f"\nRunning SAHI  slice={args.slice_size}px  overlap={args.overlap}")
sahi_results = []

for img_path in tqdm(test_imgs, desc="SAHI inference"):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    # SAHI sliced prediction
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height          = args.slice_size,
        slice_width           = args.slice_size,
        overlap_height_ratio  = args.overlap,
        overlap_width_ratio   = args.overlap,
        perform_standard_pred = True,   # also run full image
        postprocess_type      = "NMM",  # Non-Maximum Merging — better than NMS for overlaps
        verbose               = 0,
    )

    # Parse predictions — apply per-class threshold
    preds = []
    for obj in result.object_prediction_list:
        cls  = int(obj.category.id)
        conf = float(obj.score.value)
        if conf < per_class_thresh.get(cls, args.conf):
            continue
        bbox = obj.bbox
        preds.append((cls, conf, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))

    gt_boxes = load_gt_boxes(img_path.stem, img_w, img_h)
    gt_used  = [False]*len(gt_boxes)

    # Match predictions to GT
    preds_sorted = sorted(preds, key=lambda x: -x[1])
    tp=fp=fn=0
    for (pc, pconf, px1, py1, px2, py2) in preds_sorted:
        best_iou, best_gi = 0.0, -1
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gc != pc or gt_used[gi]: continue
            iou = box_iou([px1,py1,px2,py2], [gx1,gy1,gx2,gy2])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= IOU_MATCH:
            tp+=1; gt_used[best_gi]=True
        else:
            fp+=1
    fn = sum(1 for u in gt_used if not u)

    sahi_results.append({
        "image": img_path.name,
        "n_gt": len(gt_boxes), "n_pred": len(preds),
        "TP": tp, "FP": fp, "FN": fn,
    })

sahi_P, sahi_R, sahi_F1 = compute_metrics(sahi_results)

# ── Baseline (standard full-image inference) for comparison ───────────────
print("\nRunning baseline (standard inference, same thresholds) ...")
from ultralytics import YOLO
yolo_model = YOLO(str(WEIGHTS))
base_results = []

for img_path in tqdm(test_imgs, desc="Baseline inference"):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    r = yolo_model.predict(
        str(img_path), conf=default_conf, iou=args.iou_nms,
        imgsz=args.imgsz, device=args.device, verbose=False
    )[0]

    preds = []
    if r.boxes is not None and len(r.boxes):
        for i in range(len(r.boxes)):
            cls  = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            if conf < per_class_thresh.get(cls, args.conf): continue
            x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
            preds.append((cls, conf, x1, y1, x2, y2))

    gt_boxes = load_gt_boxes(img_path.stem, img_w, img_h)
    gt_used  = [False]*len(gt_boxes)
    preds_sorted = sorted(preds, key=lambda x: -x[1])
    tp=fp=fn=0
    for (pc, pconf, px1, py1, px2, py2) in preds_sorted:
        best_iou, best_gi = 0.0, -1
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gc != pc or gt_used[gi]: continue
            iou = box_iou([px1,py1,px2,py2], [gx1,gy1,gx2,gy2])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= IOU_MATCH:
            tp+=1; gt_used[best_gi]=True
        else:
            fp+=1
    fn = sum(1 for u in gt_used if not u)
    base_results.append({"image": img_path.name, "n_gt": len(gt_boxes),
                         "n_pred": len(preds), "TP": tp, "FP": fp, "FN": fn})

base_P, base_R, base_F1 = compute_metrics(base_results)

# ── Print results ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Results on test set  (IoU≥{IOU_MATCH})")
print(f"{'='*55}")
print(f"  {'Method':<30} {'P':>7} {'R':>7} {'F1':>7}")
print(f"  {'-'*51}")
print(f"  {'Standard inference':<30} {base_P:>7.4f} {base_R:>7.4f} {base_F1:>7.4f}")
print(f"  {'SAHI (slice='+str(args.slice_size)+')':<30} {sahi_P:>7.4f} {sahi_R:>7.4f} {sahi_F1:>7.4f}")
print(f"  {'Gain':<30} {sahi_P-base_P:>+7.4f} {sahi_R-base_R:>+7.4f} {sahi_F1-base_F1:>+7.4f}")
print(f"{'='*55}")

# ── Save ───────────────────────────────────────────────────────────────────
with open(OUT_DIR / "sahi_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sahi_results[0].keys())
    w.writeheader(); w.writerows(sahi_results)

summary = {
    "weights":            WEIGHTS.name,
    "slice_size":         args.slice_size,
    "overlap":            args.overlap,
    "iou_match":          IOU_MATCH,
    "used_tuned_thresh":  args.thresholds is not None,
    "base_P":  round(base_P, 4),  "base_R":  round(base_R, 4),  "base_F1":  round(base_F1, 4),
    "sahi_P":  round(sahi_P, 4),  "sahi_R":  round(sahi_R, 4),  "sahi_F1":  round(sahi_F1, 4),
    "gain_P":  round(sahi_P-base_P, 4),
    "gain_R":  round(sahi_R-base_R, 4),
    "gain_F1": round(sahi_F1-base_F1, 4),
}
with open(OUT_DIR / "sahi_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary.keys())
    w.writeheader(); w.writerow(summary)

print(f"\nSaved to: {OUT_DIR}")
print("  sahi_results.csv")
print("  sahi_summary.csv")
print("\n=== Done ===")
