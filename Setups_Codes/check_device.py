"""
check_device.py — pre-flight device + ultralytics sanity check for Exp B.

Runs three layers of checks:
  1. ultralytics.checks()    → environment summary (CUDA, torch, OS, deps)
  2. check_amp()             → does AMP actually pass on this card RIGHT NOW?
                               (the GTX 1650 Ti AMP check is flaky; need to
                               confirm before launching a long run)
  3. Real fwd+bwd pass       → loads the CBAM model + one synthetic batch at
                               batch=8 imgsz=640, measures peak VRAM, repeats
                               at batch=4 as fallback. Catches OOM before
                               training rather than 3 minutes into epoch 1.

Run:  python check_device.py
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import sys
import time
import subprocess
from pathlib import Path

import torch


# ── Make sure CBAM is registered (same gate as the training script) ─────────
def _ensure_cbam_setup():
    try:
        import ultralytics.nn.tasks as _t
        if not hasattr(_t, "CBAM"):
            raise AttributeError
    except AttributeError:
        print("[setup] CBAM not registered — running setup_cbam.py first...\n")
        subprocess.run([sys.executable, "setup_cbam.py"], check=True)
        print("\n[setup] Restarting check with patched ultralytics...\n")
        subprocess.run([sys.executable] + sys.argv, check=True)
        sys.exit(0)


_ensure_cbam_setup()


from ultralytics import YOLO
from ultralytics.utils.checks import check_amp
import ultralytics


# ── 1. Ultralytics environment check ────────────────────────────────────────
print("=" * 70)
print(" 1. ULTRALYTICS ENVIRONMENT CHECK")
print("=" * 70)
ultralytics.checks()

print()
print("=" * 70)
print(" 2. GPU MEMORY STATE")
print("=" * 70)
if not torch.cuda.is_available():
    sys.exit("CUDA not available — Exp B requires a GPU. Aborting.")

dev = torch.cuda.current_device()
props = torch.cuda.get_device_properties(dev)
free, total = torch.cuda.mem_get_info()
print(f"  Device      : {props.name}")
print(f"  Compute cap : {props.major}.{props.minor}")
print(f"  Total VRAM  : {total / 1e9:.2f} GB")
print(f"  Free VRAM   : {free / 1e9:.2f} GB")
print(f"  Driver/CUDA : {torch.version.cuda}  (PyTorch {torch.__version__})")
if free / 1e9 < 3.0:
    print("  ⚠️  Free VRAM <3GB — close Chrome/OBS/other GPU apps before training.")
else:
    print("  ✅ Free VRAM looks healthy.")


# ── 3. AMP check — the flaky part on GTX 1650 Ti ────────────────────────────
print()
print("=" * 70)
print(" 3. AMP COMPATIBILITY CHECK (the one that failed earlier)")
print("=" * 70)
# check_amp needs an actual model instance; use a fresh COCO yolo11s
try:
    probe = YOLO("yolo11s.pt").model
    amp_ok = check_amp(probe)
    if amp_ok:
        print("  ✅ AMP check passed — training will use mixed precision.")
        print("     Batch=8 should fit comfortably (~3.6 GB peak).")
    else:
        print("  ⚠️  AMP check FAILED — ultralytics will auto-disable AMP.")
        print("     Without AMP, batch=8 will likely OOM. Drop to batch=4")
        print("     in exp_B_efficient_train.py before launching.")
except Exception as e:
    print(f"  ⚠️  AMP check raised an exception: {e}")
    amp_ok = False


# ── 4. Real forward+backward pass at training config ────────────────────────
print()
print("=" * 70)
print(" 4. VRAM STRESS TEST — actual fwd+bwd at training config")
print("=" * 70)

def stress_test(batch: int, use_amp: bool) -> bool:
    """Run one fwd+bwd at given batch size; return True if it fits."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    yolo = YOLO("yolo11s_cbam.yaml")
    yolo.load("yolo11s.pt")
    model = yolo.model.to("cuda").train()

    x = torch.randn(batch, 3, 640, 640, device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    try:
        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(x)
        # Synthetic loss — sum of all output tensors (we only care about VRAM)
        if isinstance(preds, (list, tuple)):
            loss = sum(p.float().sum() for p in preds if torch.is_tensor(p))
        else:
            loss = preds.float().sum()

        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  batch={batch}, amp={use_amp}  →  ✅ FIT   "
              f"peak VRAM={peak:.2f} GB   step={elapsed:.2f}s")
        return True

    except torch.cuda.OutOfMemoryError:
        print(f"  batch={batch}, amp={use_amp}  →  ❌ OOM")
        return False
    finally:
        del model, x, optimizer
        torch.cuda.empty_cache()


# Try the planned config first, then fall back
b8_ok = stress_test(batch=8, use_amp=bool(amp_ok))
if not b8_ok:
    print("  → Falling back to batch=4...")
    stress_test(batch=4, use_amp=bool(amp_ok))


# ── 5. Verdict ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(" VERDICT")
print("=" * 70)
if b8_ok and amp_ok:
    print("  ✅ All systems go. Launch exp_B_efficient_train.py as-is.")
elif b8_ok and not amp_ok:
    print("  ⚠️  Batch=8 fits without AMP, but training is slower and val")
    print("     may OOM. Recommend dropping to batch=4 for safety.")
elif not b8_ok and amp_ok:
    print("  ⚠️  AMP works but batch=8 still OOM'd — something else is using")
    print("     VRAM. Close other GPU apps and retry.")
else:
    print("  ❌ batch=8 doesn't fit and AMP failed. Edit exp_B_efficient_train.py:")
    print("     change `batch=8` → `batch=4`, then relaunch.")
