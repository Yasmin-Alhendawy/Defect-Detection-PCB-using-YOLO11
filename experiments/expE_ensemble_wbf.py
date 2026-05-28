"""
ensemble_wbf.py — TTA + Weighted Box Fusion ensemble of Exp A and Exp B.

Pipeline (per image in val split):
  1. Run Exp A (fixed baseline) with TTA (augment=True)
  2. Run Exp B (CBAM+VFL)        with TTA (augment=True)
  3. Combine the two prediction sets via Weighted Box Fusion (WBF)
  4. Evaluate mAP@0.5 and mAP@0.5:0.95 on the WBF output

Reports the full comparison table so the paper can include:
  Exp A alone | Exp A + TTA | Exp B alone | Exp B + TTA | WBF | WBF + TTA

Why this is interesting even though Exp B underperformed:
  Exp A and Exp B make DIFFERENT mistakes (different inductive biases from CBAM
  attention + VFL re-weighting). WBF averages their box coordinates and scores,
  which often captures complementary error modes — the ensemble can beat both
  individual models even when one is weaker. Free paper result.

VRAM strategy for the GTX 1650 Ti (4 GB):
  Models are loaded SEQUENTIALLY, not concurrently. Run A on the full val set,
  cache predictions on CPU, free A. Then run B. WBF runs on CPU.

Run:
    pip install ensemble-boxes torchmetrics pycocotools  (one-time)
    python ensemble_wbf.py
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")


# ── Step 0: deps ───────────────────────────────────────────────────────────
def _pip_install(pkg: str):
    print(f"[setup] installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    from ensemble_boxes import weighted_boxes_fusion  # noqa
except ImportError:
    _pip_install("ensemble-boxes")
    from ensemble_boxes import weighted_boxes_fusion  # noqa

try:
    from torchmetrics.detection import MeanAveragePrecision  # noqa
except ImportError:
    _pip_install("torchmetrics[detection]")
    from torchmetrics.detection import MeanAveragePrecision  # noqa


# ── Step 1: ensure CBAM is registered (Exp B requires it) ──────────────────
def _ensure_cbam_setup():
    try:
        import ultralytics.nn.tasks as _t
        if not hasattr(_t, "CBAM"):
            raise AttributeError
    except AttributeError:
        print("[setup] CBAM not registered — running setup_cbam.py first...\n")
        subprocess.run([sys.executable, "setup_cbam.py"], check=True)
        print("\n[setup] Restarting with patched ultralytics...\n")
        subprocess.run([sys.executable] + sys.argv, check=True)
        sys.exit(0)


_ensure_cbam_setup()

import torch
from PIL import Image
from ultralytics import YOLO


# ── Config ──────────────────────────────────────────────────────────────────
EXP_A = "runs/detect/pcb_runs/expA_fixed_baseline/weights/best.pt"
EXP_B = "runs/detect/pcb_runs/expB_cbam_efficient-3/weights/best.pt"
DATA  = "DSPCBSD+-1/data.yaml"
IMGSZ = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"
# WBF parameters — defaults from the paper that introduced WBF (Solovyev 2021)
IOU_THR  = 0.55      # IoU threshold for grouping boxes
SKIP_BOX = 0.0       # discard boxes with score below this BEFORE WBF
# Per-model weights for the fusion — higher = trusted more
W_A = 2.0            # Exp A is stronger overall → weighted higher
W_B = 1.0            # Exp B contributes diversity


# ── Load val split paths ────────────────────────────────────────────────────
def load_val_set(data_yaml: str) -> tuple[list[Path], int]:
    """Return (image paths, num_classes) for the val split."""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    nc = cfg["nc"]
    # data.yaml uses relative paths; resolve from its parent dir
    base = Path(data_yaml).parent
    val_dir = (base / cfg["val"]).resolve()
    if not val_dir.exists():
        # Roboflow exports often use "../valid/images" — try the sibling form
        val_dir = (base / "valid" / "images").resolve()
    imgs = sorted(
        list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    )
    print(f"[data] val split: {len(imgs)} images, {nc} classes, dir={val_dir}")
    return imgs, nc


def load_gt(img_paths: list[Path]) -> list[dict]:
    """Read YOLO label files → list of {boxes_xyxy_norm, labels} in [0,1] coords."""
    targets = []
    for ip in img_paths:
        # YOLO label file lives at .../labels/<stem>.txt parallel to images
        lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        boxes, labels = [], []
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                c, cx, cy, w, h = (float(x) for x in parts[:5])
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))
        targets.append({
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,), dtype=torch.int64),
        })
    return targets


# ── Inference — runs one model over all images, returns normalized preds ────
def run_model(weights: str, img_paths: list[Path], use_tta: bool) -> list[dict]:
    """For each image return {boxes_xyxy_norm, scores, labels} on CPU."""
    print(f"\n[infer] loading {weights}  (TTA={use_tta})")
    model = YOLO(weights)
    preds = []
    for i, ip in enumerate(img_paths):
        if i % 200 == 0:
            print(f"   ... {i}/{len(img_paths)}")
        # ultralytics returns one Result per call here
        res = model.predict(
            source=str(ip),
            imgsz=IMGSZ,
            device=DEVICE,
            augment=use_tta,
            verbose=False,
            conf=0.001,         # keep wide so WBF has material to fuse
            iou=0.7,
        )[0]
        boxes_xyxy = res.boxes.xyxy.cpu()        # absolute pixel coords
        w, h = res.orig_shape[1], res.orig_shape[0]
        boxes_norm = boxes_xyxy.clone()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        boxes_norm = boxes_norm.clamp(0, 1)
        preds.append({
            "boxes":  boxes_norm,
            "scores": res.boxes.conf.cpu(),
            "labels": res.boxes.cls.cpu().to(torch.int64),
        })
    del model
    torch.cuda.empty_cache()
    return preds


# ── WBF combiner ────────────────────────────────────────────────────────────
def wbf_combine(preds_a: list[dict], preds_b: list[dict]) -> list[dict]:
    """Apply Weighted Box Fusion image-by-image, returning the same dict shape."""
    fused = []
    for pa, pb in zip(preds_a, preds_b):
        boxes_list  = [pa["boxes"].tolist(),  pb["boxes"].tolist()]
        scores_list = [pa["scores"].tolist(), pb["scores"].tolist()]
        labels_list = [pa["labels"].tolist(), pb["labels"].tolist()]
        if not boxes_list[0] and not boxes_list[1]:
            fused.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })
            continue
        b, s, l = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[W_A, W_B],
            iou_thr=IOU_THR,
            skip_box_thr=SKIP_BOX,
        )
        fused.append({
            "boxes":  torch.tensor(b, dtype=torch.float32),
            "scores": torch.tensor(s, dtype=torch.float32),
            "labels": torch.tensor(l, dtype=torch.int64),
        })
    return fused


# ── mAP evaluator ───────────────────────────────────────────────────────────
def compute_map(preds: list[dict], targets: list[dict]) -> dict:
    """Returns dict with map_50 and map_50_95 (torchmetrics convention)."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    out = metric.compute()
    return {
        "mAP@0.5":      float(out["map_50"]),
        "mAP@0.5:0.95": float(out["map"]),
        "mAR@100":      float(out["mar_100"]),
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_paths, nc = load_val_set(DATA)
    print("[data] loading ground truth labels...")
    targets = load_gt(img_paths)

    results = {}

    # --- Exp A, no TTA ---
    preds_a_plain = run_model(EXP_A, img_paths, use_tta=False)
    results["Exp A"] = compute_map(preds_a_plain, targets)

    # --- Exp A, with TTA ---
    preds_a_tta = run_model(EXP_A, img_paths, use_tta=True)
    results["Exp A + TTA"] = compute_map(preds_a_tta, targets)

    # --- Exp B, no TTA ---
    preds_b_plain = run_model(EXP_B, img_paths, use_tta=False)
    results["Exp B"] = compute_map(preds_b_plain, targets)

    # --- Exp B, with TTA ---
    preds_b_tta = run_model(EXP_B, img_paths, use_tta=True)
    results["Exp B + TTA"] = compute_map(preds_b_tta, targets)

    # --- WBF, no TTA ---
    print("\n[wbf] fusing Exp A + Exp B (no TTA)...")
    preds_wbf_plain = wbf_combine(preds_a_plain, preds_b_plain)
    results["WBF (A+B)"] = compute_map(preds_wbf_plain, targets)

    # --- WBF, with TTA ---
    print("[wbf] fusing Exp A + Exp B (both with TTA)...")
    preds_wbf_tta = wbf_combine(preds_a_tta, preds_b_tta)
    results["WBF + TTA"] = compute_map(preds_wbf_tta, targets)

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ENSEMBLE RESULTS — val split")
    print("=" * 70)
    print(f"  WBF weights:  Exp A = {W_A},  Exp B = {W_B}")
    print(f"  IoU thr = {IOU_THR},  skip_box_thr = {SKIP_BOX}")
    print()
    print(f"  {'Config':<20} {'mAP@0.5':>10} {'mAP@0.5:0.95':>15} {'mAR@100':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*10}")
    for name, m in results.items():
        print(f"  {name:<20} {m['mAP@0.5']:>10.4f} "
              f"{m['mAP@0.5:0.95']:>15.4f} {m['mAR@100']:>10.4f}")

    # Save WBF+TTA predictions to disk for downstream per-class threshold tuning
    out_pt = Path("ensemble_wbf_tta_preds.pt")
    torch.save(
        {"preds": preds_wbf_tta, "targets": targets, "img_paths": [str(p) for p in img_paths]},
        out_pt,
    )
    print(f"\n[save] WBF+TTA predictions saved → {out_pt}")
    print("       Use this file for per-class threshold tuning next.")

    # Also save the summary table as csv for the paper
    import csv
    summary_csv = Path("ensemble_wbf_summary.csv")
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mAP@0.5", "mAP@0.5:0.95", "mAR@100"])
        for name, m in results.items():
            w.writerow([name, f"{m['mAP@0.5']:.4f}",
                              f"{m['mAP@0.5:0.95']:.4f}",
                              f"{m['mAR@100']:.4f}"])
    print(f"[save] summary table → {summary_csv}")
