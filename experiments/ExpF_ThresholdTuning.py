"""
Exp F — Per-Class Confidence Threshold Tuning
==============================================
Problem: YOLO's default conf=0.25 is a one-size-fits-all threshold.
With 9 imbalanced defect classes, each class has a different
precision/recall trade-off. Tuning per-class thresholds on the val set
can recover mAP without any retraining.

How it works:
  1. Run inference on val set with conf=0.001 (collect ALL predictions)
  2. For each class: sweep thresholds 0.05→0.95, find threshold that
     maximises F1 on val
  3. Apply optimal per-class thresholds to test set
  4. Compare: default conf=0.25 vs tuned thresholds

Usage:
  # Tune on Exp A weights (default)
  python experiments/ExpF_ThresholdTuning.py

  # Tune on a specific weights file
  python experiments/ExpF_ThresholdTuning.py --weights runs/detect/pcb_runs/expB_cbam_pretrained/weights/best.pt

Output (saved alongside weights in the run folder):
  threshold_tuning_results.csv   — per-class optimal thresholds + F1 gains
  threshold_tuning_summary.csv   — overall before/after metrics for paper table
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=
    "runs/detect/pcb_runs/expA_fixed_baseline/weights/best.pt",
    help="Path to YOLO weights file")
parser.add_argument("--data", default="DSPCBSD+-1/data.yaml")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--device", default="0")
args = parser.parse_args()

WEIGHTS   = Path(args.weights)
DATA      = args.data
IMGSZ     = args.imgsz
DEVICE    = args.device
NC        = 9
THRESH_SWEEP = np.arange(0.05, 0.96, 0.05).tolist()
IOU_MATCH = 0.5     # IoU to count a detection as TP

# Output goes next to the weights file
OUT_DIR = WEIGHTS.parent.parent   # run folder (e.g. expA_fixed_baseline/)
print(f"Weights : {WEIGHTS}")
print(f"Output  : {OUT_DIR}")

# ─── Load model ───────────────────────────────────────────────────────────────
model = YOLO(str(WEIGHTS))

# ─── Helper: IoU between two boxes [x1,y1,x2,y2] ─────────────────────────────
def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ─── Collect raw predictions + GT on a split ──────────────────────────────────
def collect_predictions(split="val"):
    """
    Returns:
      preds: dict[img_stem] = list of (cls, conf, x1, y1, x2, y2)
      gts:   dict[img_stem] = list of (cls, x1, y1, x2, y2)
    """
    import yaml
    with open(DATA) as f:
        cfg = yaml.safe_load(f)

    # Resolve image directory
    # Roboflow data.yaml uses paths like "../valid/images" even though
    # valid/ sits INSIDE the dataset folder — strip the leading "../"
    data_root = Path(DATA).parent
    rel = cfg[split].lstrip("./\\").replace("../", "")   # "valid/images"
    img_dir = (data_root / rel).resolve()

    img_paths = sorted(img_dir.glob("*.jpg"))
    lbl_dir = img_dir.parent / "labels"   # sibling of images/

    preds = {}
    gts   = {}

    # Run inference at very low conf to capture everything
    results = model.predict(
        source=str(img_dir),
        conf=0.001,
        iou=0.45,          # NMS IoU — keep loose at this stage
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
        stream=True,
    )

    for r in tqdm(results, total=len(img_paths), desc=f"Predicting {split}"):
        stem = Path(r.path).stem
        boxes = r.boxes

        # Predictions
        p_list = []
        if boxes is not None and len(boxes):
            xyxy  = boxes.xyxy.cpu().numpy()   # type: ignore # (N,4)
            confs = boxes.conf.cpu().numpy()   # type: ignore # (N,)
            clss  = boxes.cls.cpu().numpy().astype(int)  # type: ignore # (N,)
            for i in range(len(clss)):
                p_list.append((clss[i], float(confs[i]),
                               *xyxy[i].tolist()))
        preds[stem] = p_list

        # Ground truth
        lbl_path = lbl_dir / (stem + ".txt")
        g_list = []
        if lbl_path.exists():
            ih, iw = r.orig_shape
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw/2) * iw; y1 = (cy - bh/2) * ih
                    x2 = (cx + bw/2) * iw; y2 = (cy + bh/2) * ih
                    g_list.append((cls, x1, y1, x2, y2))
        gts[stem] = g_list

    return preds, gts


# ─── Compute TP/FP/FN for a given set of per-class thresholds ─────────────────
def evaluate(preds, gts, thresholds_per_class):
    """
    thresholds_per_class: list of length NC, one threshold per class
    Returns: per-class precision, recall, F1 + micro-averaged
    """
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)

    for stem, pred_list in preds.items():
        gt_list   = gts.get(stem, [])
        gt_used   = [False] * len(gt_list)

        # Filter predictions by per-class threshold
        filtered  = [(c, conf, x1, y1, x2, y2)
                     for (c, conf, x1, y1, x2, y2) in pred_list
                     if conf >= thresholds_per_class[c]]

        # Sort by descending confidence
        filtered.sort(key=lambda x: -x[1])

        for (pc, pconf, px1, py1, px2, py2) in filtered:
            # Find best matching GT of same class
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_list):
                if gc != pc or gt_used[gi]:
                    continue
                iou_val = box_iou([px1,py1,px2,py2], [gx1,gy1,gx2,gy2])
                if iou_val > best_iou:
                    best_iou, best_gi = iou_val, gi

            if best_iou >= IOU_MATCH:
                TP[pc] += 1
                gt_used[best_gi] = True
            else:
                FP[pc] += 1

        # Count FN
        for gi, (gc, *_) in enumerate(gt_list):
            if not gt_used[gi]:
                FN[gc] += 1

    # Per-class metrics
    per_class = {}
    for c in range(NC):
        tp, fp, fn = TP[c], FP[c], FN[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "TP": tp, "FP": fp, "FN": fn}

    # Micro-average
    tp_tot = sum(TP.values()); fp_tot = sum(FP.values()); fn_tot = sum(FN.values())
    micro_prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    micro_rec  = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    micro_f1   = 2*micro_prec*micro_rec / (micro_prec+micro_rec) if (micro_prec+micro_rec) > 0 else 0.0

    return per_class, {"precision": micro_prec, "recall": micro_rec,
                       "f1": micro_f1, "TP": tp_tot, "FP": fp_tot, "FN": fn_tot}


# ─── Step 1: Collect val predictions ─────────────────────────────────────────
print("\n=== Step 1: Collecting val predictions (conf=0.001) ===")
val_preds, val_gts = collect_predictions("val")

total_gt  = sum(len(v) for v in val_gts.values())
total_det = sum(len(v) for v in val_preds.values())
print(f"  Val images   : {len(val_preds)}")
print(f"  GT boxes     : {total_gt}")
print(f"  Raw preds    : {total_det}  (before threshold filtering)")

# ─── Step 2: Baseline — default conf=0.25 for all classes ────────────────────
print("\n=== Step 2: Baseline (conf=0.25 all classes) ===")
default_thresh = [0.25] * NC
base_per_class, base_micro = evaluate(val_preds, val_gts, default_thresh)
print(f"  Micro  P={base_micro['precision']:.4f}  R={base_micro['recall']:.4f}  F1={base_micro['f1']:.4f}")
for c in range(NC):
    m = base_per_class[c]
    print(f"  cls {c}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"(TP={m['TP']} FP={m['FP']} FN={m['FN']})")

# ─── Step 3: Sweep per-class thresholds on val ───────────────────────────────
print("\n=== Step 3: Sweeping thresholds per class ===")
optimal_thresholds = [0.25] * NC   # start from default

for c in range(NC):
    best_f1  = base_per_class[c]["f1"]
    best_thr = 0.25

    for thr in THRESH_SWEEP:
        # Only change class c's threshold; keep others at current best
        trial_thresh = optimal_thresholds.copy()
        trial_thresh[c] = thr
        pc, _ = evaluate(val_preds, val_gts, trial_thresh)
        if pc[c]["f1"] > best_f1:
            best_f1  = pc[c]["f1"]
            best_thr = thr

    optimal_thresholds[c] = best_thr
    gain = best_f1 - base_per_class[c]["f1"]
    print(f"  cls {c}: thresh {0.25:.2f} → {best_thr:.2f}  "
          f"F1 {base_per_class[c]['f1']:.4f} → {best_f1:.4f}  "
          f"(+{gain:.4f})")

# ─── Step 4: Evaluate tuned thresholds on val ────────────────────────────────
print("\n=== Step 4: Tuned thresholds on val ===")
tuned_per_class, tuned_micro = evaluate(val_preds, val_gts, optimal_thresholds)
print(f"  Micro  P={tuned_micro['precision']:.4f}  R={tuned_micro['recall']:.4f}  F1={tuned_micro['f1']:.4f}")
print(f"  F1 gain vs baseline: {tuned_micro['f1'] - base_micro['f1']:+.4f}")

# ─── Step 5: Apply to test set ────────────────────────────────────────────────
print("\n=== Step 5: Applying tuned thresholds to test set ===")
test_preds, test_gts = collect_predictions("test")

_, test_base_micro  = evaluate(test_preds, test_gts, [0.25]*NC)
_, test_tuned_micro = evaluate(test_preds, test_gts, optimal_thresholds)

print(f"  Test — default  conf=0.25 : P={test_base_micro['precision']:.4f}  "
      f"R={test_base_micro['recall']:.4f}  F1={test_base_micro['f1']:.4f}")
print(f"  Test — tuned per-class    : P={test_tuned_micro['precision']:.4f}  "
      f"R={test_tuned_micro['recall']:.4f}  F1={test_tuned_micro['f1']:.4f}")
print(f"  F1 gain on test: {test_tuned_micro['f1'] - test_base_micro['f1']:+.4f}")

# ─── Save results ─────────────────────────────────────────────────────────────
# Per-class thresholds + val F1 change
rows = []
for c in range(NC):
    rows.append({
        "class":           c,
        "default_thresh":  0.25,
        "tuned_thresh":    round(optimal_thresholds[c], 2),
        "val_f1_baseline": round(base_per_class[c]["f1"], 4),
        "val_f1_tuned":    round(tuned_per_class[c]["f1"], 4),
        "val_f1_gain":     round(tuned_per_class[c]["f1"] - base_per_class[c]["f1"], 4),
    })

with open(OUT_DIR / "threshold_tuning_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Summary for paper table
summary = {
    "weights":              str(WEIGHTS.name),
    "iou_match":            IOU_MATCH,
    # val
    "val_default_P":        round(base_micro["precision"], 4),
    "val_default_R":        round(base_micro["recall"], 4),
    "val_default_F1":       round(base_micro["f1"], 4),
    "val_tuned_P":          round(tuned_micro["precision"], 4),
    "val_tuned_R":          round(tuned_micro["recall"], 4),
    "val_tuned_F1":         round(tuned_micro["f1"], 4),
    # test
    "test_default_P":       round(test_base_micro["precision"], 4),
    "test_default_R":       round(test_base_micro["recall"], 4),
    "test_default_F1":      round(test_base_micro["f1"], 4),
    "test_tuned_P":         round(test_tuned_micro["precision"], 4),
    "test_tuned_R":         round(test_tuned_micro["recall"], 4),
    "test_tuned_F1":        round(test_tuned_micro["f1"], 4),
    # optimal thresholds (json string)
    "optimal_thresholds":   json.dumps({str(c): round(t, 2)
                                        for c, t in enumerate(optimal_thresholds)}),
}
with open(OUT_DIR / "threshold_tuning_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary.keys())
    writer.writeheader()
    writer.writerow(summary)

# Also save thresholds as standalone JSON for use in other scripts
with open(OUT_DIR / "optimal_thresholds.json", "w") as f:
    json.dump({str(c): round(t, 2) for c, t in enumerate(optimal_thresholds)}, f, indent=2)

print(f"\nSaved to: {OUT_DIR}")
print(f"  threshold_tuning_results.csv")
print(f"  threshold_tuning_summary.csv")
print(f"  optimal_thresholds.json  ← use this in SAHI + ensemble scripts")
print("\n=== Done ===")
