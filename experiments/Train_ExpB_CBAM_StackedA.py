"""
Train_ExpB_CBAM_StackedA.py — Exp B: YOLOv11s + CBAM, fine-tuned from Exp A weights

Strategy:
  - Architecture : yolo11s_cbam.yaml (CBAM @ P3/P4/P5)
  - Init         : Exp A best.pt (partial transfer — backbone + head match,
                   CBAM modules init from scratch via xavier)
  - Data         : oversampled train set (6628 imgs, classes 6&7 doubled)
  - Epochs       : 25  (domain already learned; CBAM just needs to find attention)
  - LR           : 0.001  (fine-tuning — preserve Exp A backbone weights)

Why this beats expB_cbam_efficient:
  - Efficient used COCO yolo11s.pt → had to re-learn PCB domain from scratch
  - This uses Exp A → backbone already knows PCB defects after 100 epochs
  - CBAM converges in ~20 epochs instead of ~60+

Expected outcome: mAP@0.5 ≥ Exp A (0.8388), with better class 6 & 7 F1
"""

import os
import sys
import subprocess
from pathlib import Path

# ── env tuning BEFORE torch/ultralytics import ─────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch


# ── Ensure CBAM is registered ───────────────────────────────────────────────
def _ensure_cbam():
    try:
        import ultralytics.nn.tasks as _t
        if not hasattr(_t, "CBAM"):
            raise AttributeError
    except AttributeError:
        print("[setup] CBAM not registered — running setup_cbam.py...\n")
        setup = Path(__file__).parent / "setup_cbam.py"
        subprocess.run([sys.executable, str(setup)], check=True)
        print("\n[setup] Restarting...\n")
        subprocess.run([sys.executable] + sys.argv, check=True)
        sys.exit(0)


_ensure_cbam()

from ultralytics import YOLO  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────────
EXP_A_WEIGHTS = Path("runs/detect/pcb_runs/expA_fixed_baseline/weights/best.pt")
CBAM_YAML     = Path("yolo11s_cbam.yaml")
RUN_DIR       = Path("pcb_runs/expB_cbam_stackedA")
LAST_CKPT     = RUN_DIR / "weights" / "last.pt"


def build_model() -> YOLO:
    if LAST_CKPT.exists():
        print(f"[resume] Resuming from {LAST_CKPT}\n")
        return YOLO(str(LAST_CKPT))

    if not EXP_A_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Exp A weights not found at {EXP_A_WEIGHTS}\n"
            "Run Train_Exp_A_FixedBaseline.py first."
        )

    print(f"[init] Loading CBAM architecture from {CBAM_YAML}")
    model = YOLO(str(CBAM_YAML))

    print(f"[init] Transferring Exp A weights from {EXP_A_WEIGHTS}")
    model.load(str(EXP_A_WEIGHTS))
    # model.load() copies all matching layer weights (Conv, C3k2, SPPF,
    # C2PSA, detection head) and leaves CBAM modules at xavier init.
    print("[init] Weight transfer done. CBAM modules will train from scratch.\n")

    return model


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {torch.cuda.get_device_name(0)} | "
              f"free={free/1e9:.2f} GB / total={total/1e9:.2f} GB\n")

    model     = build_model()
    is_resume = LAST_CKPT.exists()

    model.train(
        # ── Core ──────────────────────────────────────────────────────────
        data    = "DSPCBSD+-1/data.yaml",   # includes oversampled train set
        epochs  = 25,
        batch   = 8,
        imgsz   = 640,
        device  = "0",

        # ── Naming ────────────────────────────────────────────────────────
        project  = "pcb_runs",
        name     = "expB_cbam_stackedA",
        exist_ok = is_resume,
        resume   = is_resume,

        # ── Fine-tuning LR (lower than Exp A's 0.01) ──────────────────────
        # Backbone weights are already good — don't overwrite them aggressively
        cos_lr        = True,
        lr0           = 0.001,      # 10× lower than Exp A
        lrf           = 0.01,
        warmup_epochs = 1.0,        # short warmup — already converged backbone
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.05,

        # ── Loss — identical to Exp A (no VFL; clean ablation) ────────────
        cls = 0.5,
        box = 7.5,
        dfl = 1.5,

        # ── Training schedule — identical to Exp A ─────────────────────────
        close_mosaic = 20,
        erasing      = 0.0,
        patience     = 15,          # stop if no improvement for 15 epochs
        optimizer    = "auto",
        weight_decay = 0.0005,
        momentum     = 0.937,

        # ── Augmentation — identical to Exp A ─────────────────────────────
        auto_augment = "randaugment",
        fliplr       = 0.5,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        translate    = 0.1,
        scale        = 0.5,
        mosaic       = 1.0,

        # ── Infrastructure ────────────────────────────────────────────────
        amp         = True,
        cache       = "disk",
        workers     = 2,
        seed        = 0,
        deterministic = True,
        pretrained  = False,        # weights loaded manually above
        save        = True,
        save_period = -1,
        plots       = False,        # final val generates paper figures
        val         = True,
        verbose     = True,
    )

    # ── Final validation with plots for paper ─────────────────────────────
    print("\n[post] Running final validation with plots...\n")
    best = RUN_DIR / "weights" / "best.pt"
    if best.exists():
        final = YOLO(str(best))
        final.val(
            data    = "DSPCBSD+-1/data.yaml",
            imgsz   = 640,
            batch   = 8,
            device  = "0",
            plots   = True,
            project = "pcb_runs",
            name    = "expB_cbam_stackedA_val",
            split   = "val",
        )
    else:
        print(f"[post] best.pt not found — skipping final val.")
