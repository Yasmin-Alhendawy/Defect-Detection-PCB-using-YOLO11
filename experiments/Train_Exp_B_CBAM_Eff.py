"""
exp_B_efficient_train.py — Exp B: YOLOv11s + CBAM + VFL (efficiency-tuned, INDEPENDENT)

Clean ablation against Exp A: identical init (COCO yolo11s.pt), identical
epoch budget, identical schedule — the ONLY difference vs Exp A is the CBAM
attention modules + Varifocal Loss. This is the comparison the paper needs.

Architecture : YOLOv11s backbone + CBAM attention @ P3/P4/P5
Init         : COCO pretrained yolo11s.pt (NOT Exp A weights — clean ablation).
               Backbone weights load into CBAM yaml; CBAM modules init from scratch.
Loss         : Varifocal Loss (replaces BCE) — handles 6-class imbalance.
Epochs       : 50   (tests efficiency — same budget so any mAP delta = CBAM, not training time)
Patience     : 10   (aggressive early-stop — save GPU time if plateau hits)
Batch        : 8    (GTX 1650 Ti 4 GB VRAM limit)

What changed vs v2:
  - No warm-start from Exp A → starts from yolo11s.pt like Exp A did
  - epochs 50 → 100, lr0 0.002 → 0.01, warmup 1.0 → 3.0 (matches Exp A schedule)
  - patience 30 → 10 (aggressive — stop within 10 epochs of no val improvement)
  - All efficiency knobs from v2-efficient retained (cache="ram", cudnn.benchmark, etc.)

⚠️ Note on patience=10 + CBAM:
  CBAM modules init from scratch and can be slow starters — the first 5–15 epochs
  often show flat/noisy val mAP while attention layers find their footing. If the
  run stops at ~epoch 15 with bad mAP, consider bumping patience back to 20.

Run order:
    1. python setup_cbam.py     (one-time; patches .venv for CBAM)
    2. python exp_B_efficient_train.py
"""

import os
import sys
import subprocess
from pathlib import Path

# ── Step -1: env tuning MUST happen before torch / ultralytics import ──────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
# NOTE: don't set YOLO_PERSISTENT_WORKERS on Windows — it amplifies spawn
# memory cost (every worker re-pickles the parent process)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Step 0: ensure CBAM is registered ──────────────────────────────────────
def _ensure_cbam_setup():
    try:
        import ultralytics.nn.tasks as _t
        if not hasattr(_t, "CBAM"):
            raise AttributeError
    except AttributeError:
        print("[setup] CBAM not registered — running setup_cbam.py first...\n")
        setup_script = Path(__file__).parent / "setup_cbam.py"
        subprocess.run([sys.executable, str(setup_script)], check=True)
        print("\n[setup] Restarting with patched ultralytics...\n")
        subprocess.run([sys.executable] + sys.argv, check=True)
        sys.exit(0)


_ensure_cbam_setup()

# ── Imports (after CBAM is registered) ─────────────────────────────────────
from ultralytics import YOLO  # noqa: E402


# ── cuDNN autotune — safe for fixed input size (imgsz=640, batch=8) ─────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Varifocal Loss ──────────────────────────────────────────────────────────
class VFLAsBCE(nn.Module):
    """Varifocal Loss matching BCEWithLogitsLoss(reduction='none') interface."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        label = (gt_score > 0).float()
        weight = (
            self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label)
            + gt_score * label
        )
        return F.binary_cross_entropy_with_logits(
            pred_score.float(), gt_score.float(), reduction="none"
        ) * weight


def patch_vfl(model: YOLO) -> None:
    _orig = model.model.init_criterion # type: ignore

    def _patched():
        criterion = _orig()
        criterion.bce = VFLAsBCE()
        print("[VFL] Varifocal Loss active — classification BCE replaced.\n")
        return criterion

    model.model.init_criterion = _patched  # type: ignore


# ── Resume detection ────────────────────────────────────────────────────────
RUN_DIR   = Path("pcb_runs/expB_cbam_efficient")
LAST_CKPT = RUN_DIR / "weights" / "last.pt"


def _build_model() -> YOLO:
    """If an interrupted run exists, resume from last.pt; else fresh COCO init."""
    if LAST_CKPT.exists():
        print(f"[resume] Found {LAST_CKPT} — resuming interrupted run.\n")
        return YOLO(str(LAST_CKPT))

    # Independent of Exp A — load COCO weights into the CBAM architecture.
    # Backbone/head weights transfer; CBAM modules init from scratch (xavier).
    model = YOLO("yolo11s_cbam.yaml")
    model.load("yolo11s.pt")
    patch_vfl(model)
    return model


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {torch.cuda.get_device_name(0)} | "
              f"free={free / 1e9:.2f} GB / total={total / 1e9:.2f} GB\n")

    model = _build_model()
    is_resume = LAST_CKPT.exists()

    model.train(
        # ── Core ──────────────────────────────────────────────
        data        = "DSPCBSD+-1/data.yaml",
        epochs      = 50,           # tests
        batch       = 8,
        imgsz       = 640,
        device      = "0",

        # ── Naming ────────────────────────────────────────────
        project     = "pcb_runs",
        name        = "expB_cbam_efficient",
        exist_ok    = is_resume,
        resume      = is_resume,

        # ── Loss & augmentation (matches Exp A, P1 fixes) ─────
        cls         = 0.5,
        close_mosaic= 10,
        erasing     = 0.0,

        # ── LR — scratch schedule (matches Exp A) ─────────────
        cos_lr      = True,
        lr0         = 0.01,          # was 0.002 (warm-start) → 0.01 (scratch, = Exp A)
        lrf         = 0.01,
        patience    = 10,            # aggressive early-stop

        # ── Optimizer & warmup (matches Exp A) ────────────────
        optimizer       = "auto",
        weight_decay    = 0.0005,
        momentum        = 0.937,
        warmup_epochs   = 3.0,       # was 1.0 (warm-start) → 3.0 (scratch, = Exp A)
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,       # was 0.05 → 0.1 (matches Exp A)

        # ── Loss weights ──────────────────────────────────────
        box         = 7.5,
        dfl         = 1.5,

        # ── Augmentation (matches Exp A) ──────────────────────
        auto_augment= "randaugment",
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        translate   = 0.1,
        scale       = 0.5,
        mosaic      = 1.0,

        # ── EFFICIENCY KNOBS ──────────────────────────────────
        amp         = True,          # ultralytics auto-disables on GTX 1650 Ti
                                     # if its check fails — if it does, drop
                                     # batch to 4 to avoid OOM
        cache       = "disk",        # was "ram" — only 6.7GB free on host & non-
                                     # deterministic. disk = ~same speedup, fits.
        workers     = 2,             # was 4 — Windows spawn pickles parent state
                                     # per worker; 4 workers OOM'd on this host
        save_period = -1,            # only best.pt + last.pt
        plots       = False,         # skip per-epoch plots (final val regenerates)
        val         = True,
        verbose     = True,

        # ── Reproducibility ───────────────────────────────────
        seed            = 0,
        deterministic   = True,
        pretrained      = False,     # weights loaded manually via model.load()
        save            = True,
    )

    # ── Final validation w/ plots for paper figures ─────────────────────────
    print("\n[post] Running final validation with plots ON for paper figures...\n")
    best_ckpt = RUN_DIR / "weights" / "best.pt"
    if best_ckpt.exists():
        final = YOLO(str(best_ckpt))
        final.val(
            data    = "DSPCBSD+-1/data.yaml",
            imgsz   = 640,
            batch   = 8,
            device  = "0",
            plots   = True,
            project = "pcb_runs",
            name    = "expB_cbam_efficient_final_val",
            split   = "val",
        )
    else:
        print(f"[post] best.pt not found at {best_ckpt} — skipping final val.")
