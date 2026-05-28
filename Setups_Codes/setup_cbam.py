"""
setup_cbam.py — One-time patcher that registers CBAM with Ultralytics YOLOv11.

What it does
------------
1. Copies cbam_module.py into the active .venv's site-packages so it is
   importable anywhere in the environment.
2. Patches ultralytics/nn/tasks.py to:
     a) Import CBAM at module load time (so globals()[m] resolves 'CBAM').
     b) Add CBAM to the base_modules frozenset so parse_model handles
        channel width-scaling correctly (c1, c2 = ch[f], args[0]).

Run once before training:
    python setup_cbam.py

Safe to re-run — idempotent (checks for existing patch markers).
"""

import sys
import shutil
from pathlib import Path


# ── Locate .venv site-packages ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR   = SCRIPT_DIR / ".venv"

def find_site_packages() -> Path:
    candidates = list(VENV_DIR.glob("Lib/site-packages")) + \
                 list(VENV_DIR.glob("lib/python*/site-packages"))
    if not candidates:
        raise RuntimeError(
            f"Could not find site-packages inside {VENV_DIR}.\n"
            "Make sure you are running from the DSPCBSD project root."
        )
    return candidates[0]


# ── Step 1: copy cbam_module.py into site-packages ─────────────────────────
def install_cbam_module(site_packages: Path) -> None:
    src = SCRIPT_DIR / "cbam_module.py"
    dst = site_packages / "cbam_module.py"
    if not src.exists():
        raise FileNotFoundError(f"cbam_module.py not found at {src}")
    shutil.copy2(src, dst)
    print(f"  [✓] cbam_module.py → {dst}")


# ── Step 2: patch ultralytics/nn/tasks.py ──────────────────────────────────
PATCH_MARKER_IMPORT  = "# <<CBAM-PATCH-IMPORT>>"
PATCH_MARKER_FROZENSET = "# <<CBAM-PATCH-FROZENSET>>"

IMPORT_PATCH = f"""\
{PATCH_MARKER_IMPORT}
try:
    from cbam_module import CBAM  # noqa: F401 — CBAM registered in globals for YAML parsing
except ImportError as _e:
    raise ImportError(
        "cbam_module not found. Run setup_cbam.py first."
    ) from _e
"""

def patch_tasks_py(site_packages: Path) -> None:
    tasks_path = site_packages / "ultralytics" / "nn" / "tasks.py"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.py not found at {tasks_path}")

    text = tasks_path.read_text(encoding="utf-8")

    # ── a) Add import after existing ultralytics.nn.modules imports ─────────
    if PATCH_MARKER_IMPORT in text:
        print("  [=] Import patch already applied — skipping.")
    else:
        # Insert right before: 'from ultralytics.nn.autobackend import'
        anchor = "from ultralytics.nn.autobackend import"
        if anchor not in text:
            raise RuntimeError(f"Could not find anchor '{anchor}' in tasks.py")
        text = text.replace(anchor, IMPORT_PATCH + "\n" + anchor, 1)
        print("  [✓] Import patch applied.")

    # ── b) Add CBAM to base_modules frozenset ───────────────────────────────
    if PATCH_MARKER_FROZENSET in text:
        print("  [=] base_modules patch already applied — skipping.")
    else:
        # Find the closing brace of base_modules frozenset — it ends with '        }\n    )'
        # We look for '            A2C2f,' (last entry) and add CBAM after it
        old_entry = "            A2C2f,\n        }"
        new_entry = (
            f"            A2C2f,\n"
            f"            CBAM,  {PATCH_MARKER_FROZENSET}\n"
            f"        }}"
        )
        if old_entry not in text:
            raise RuntimeError(
                "Could not find 'A2C2f,' closing entry in base_modules frozenset.\n"
                "The ultralytics version may have changed. Check tasks.py manually."
            )
        text = text.replace(old_entry, new_entry, 1)
        print("  [✓] base_modules patch applied.")

    tasks_path.write_text(text, encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Setting up CBAM for Ultralytics YOLOv11")
    print("=" * 60)

    site_packages = find_site_packages()
    print(f"\nSite-packages: {site_packages}\n")

    print("Step 1 — Installing cbam_module.py into site-packages...")
    install_cbam_module(site_packages)

    print("\nStep 2 — Patching ultralytics/nn/tasks.py...")
    patch_tasks_py(site_packages)

    print("\n" + "=" * 60)
    print("  CBAM setup complete. You can now train with yolo11s_cbam.yaml.")
    print("=" * 60)

    # Quick smoke-test
    print("\nSmoke-testing CBAM import...")
    try:
        from cbam_module import CBAM  # noqa: F401
        import torch
        x = torch.zeros(1, 256, 40, 40)
        m = CBAM(256)
        out = m(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print("  [✓] CBAM forward pass OK — shape preserved.")
    except Exception as e:
        print(f"  [!] Smoke test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
