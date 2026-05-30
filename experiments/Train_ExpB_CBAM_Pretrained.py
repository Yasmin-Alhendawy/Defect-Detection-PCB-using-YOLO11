"""
Exp B (Fixed) — YOLOv11s + CBAM with Pretrained Backbone Weights
=================================================================
The Problem:
  Previous expB runs used pretrained=False because yolo11s_cbam.yaml has
  CBAM layers inserted, so yolo11s.pt can't be loaded directly (shape mismatch).
  Training from scratch = slow convergence = CBAM looks worse than baseline.

The Fix — Partial Weight Loading:
  1. Load CBAM model from yaml (randomly initialised)
  2. Load standard yolo11s.pt weights
  3. Copy every backbone layer whose weights MATCH between the two models
     (CBAM layers stay randomly initialised — they'll learn from a good backbone)
  4. Train normally from this warm start

Backbone layer mapping (standard index → CBAM index):
  Standard:  0  1  2  3  4  5  6  7  8  9  10
  CBAM:      0  1  2  3  4  6  7  9 10 12  13
  (Gaps are CBAM attention layers: 5, 8, 11 — randomly initialised, fast to learn)

Run:
    python experiments/Train_ExpB_CBAM_Pretrained.py
"""

import re
import sys
from pathlib import Path

# ── Ensure CBAM is importable (setup_cbam must have been run at least once) ──
try:
    from cbam_module import CBAM  # noqa: F401
except ImportError:
    print("ERROR: CBAM not found. Run this first:")
    print("  python Setups_Codes/setup_cbam.py")
    sys.exit(1)

import torch
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
WEIGHTS  = BASE / "yolo11s.pt"
YAML     = BASE / "yolo11s_cbam.yaml"
DATA     = "DSPCBSD+-1/data.yaml"

assert WEIGHTS.exists(), f"yolo11s.pt not found at {WEIGHTS}"
assert YAML.exists(),    f"yolo11s_cbam.yaml not found at {YAML}"

# ── Backbone layer index mapping: standard → CBAM ──────────────────────────
# Standard YOLOv11s layers 0-10 map to these CBAM layer indices.
# CBAM layers 5, 8, 11 (attention modules) are NOT in standard → skip them.
BACKBONE_MAP = {
    0:  0,   # Conv  P1/2
    1:  1,   # Conv  P2/4
    2:  2,   # C3k2
    3:  3,   # Conv  P3/8
    4:  4,   # C3k2  P3 features
    5:  6,   # Conv  P4/16   (skips CBAM@5)
    6:  7,   # C3k2  P4 features
    7:  9,   # Conv  P5/32   (skips CBAM@8)
    8: 10,   # C3k2  P5 features
    9: 12,   # SPPF           (skips CBAM@11)
   10: 13,   # C2PSA
}

if __name__ == "__main__":
 # ── Step 1: Load both models ─────────────────────────────────────────────
 print("Loading standard yolo11s.pt ...")
 std_model  = YOLO(str(WEIGHTS))
 std_sd     = std_model.model.state_dict() # type: ignore

 print("Building CBAM model from yaml ...")
 cbam_model = YOLO(str(YAML))
 cbam_sd    = cbam_model.model.state_dict() # type: ignore

 # ── Step 2: Build remapped state dict ───────────────────────────────────
 print("\nTransferring backbone weights ...")
 transferred = 0
 skipped     = 0
 new_sd      = {k: v.clone() for k, v in cbam_sd.items()}

 for std_key, std_val in std_sd.items():
     m = re.match(r"^model\.(\d+)\.(.*)", std_key)
     if not m:
         continue
     std_idx = int(m.group(1))
     rest    = m.group(2)

     cbam_idx = BACKBONE_MAP.get(std_idx)
     if cbam_idx is None:
         continue

     cbam_key = f"model.{cbam_idx}.{rest}"
     if cbam_key not in cbam_sd:
         skipped += 1
         continue

     if cbam_sd[cbam_key].shape != std_val.shape:
         skipped += 1
         continue

     new_sd[cbam_key] = std_val.clone()
     transferred += 1

 cbam_model.model.load_state_dict(new_sd, strict=False) # type: ignore
 print(f"  Transferred : {transferred} tensors")
 print(f"  Skipped     : {skipped} tensors (shape mismatch or head layer)")
 print(f"  CBAM-only   : {len(cbam_sd) - transferred - skipped} tensors (randomly initialised)")

 # ── Step 3: Train ────────────────────────────────────────────────────────
 print("\n=== Starting Exp B training (pretrained backbone) ===\n")

 cbam_model.train(
     data        = DATA,
     epochs      = 50,
     patience    = 15,
     batch       = 8,
     imgsz       = 640,
     device      = "0",
     workers     = 4,
     amp         = True,
     cos_lr      = True,
     close_mosaic= 20,
     warmup_epochs = 3,
     lr0         = 0.005,
     lrf         = 0.01,
     weight_decay= 0.0005,
     cls         = 0.5,
     auto_augment= "randaugment",
     erasing     = 0.0,
     project     = "pcb_runs",
     name        = "expB_cbam_pretrained",
     exist_ok    = True,
     seed        = 42,
     pretrained  = False,
     freeze      = None,
     verbose     = True,
 )

 print("\n=== Training complete ===")
 print("Weights saved to: pcb_runs/expB_cbam_pretrained/weights/best.pt")
 print("\nNext steps:")
 print("  1. python experiments/ExpF_ThresholdTuning.py --weights runs/detect/pcb_runs/expB_cbam_pretrained/weights/best.pt")
 print("  2. python experiments/ExpD_SAHI.py --weights runs/detect/pcb_runs/expB_cbam_pretrained/weights/best.pt")
