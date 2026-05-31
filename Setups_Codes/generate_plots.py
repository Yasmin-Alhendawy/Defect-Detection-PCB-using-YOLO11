"""
generate_plots.py — Run validation with plots for any experiment weights.
Usage: python experiments/generate_plots.py
"""
from ultralytics import YOLO

WEIGHTS = "runs/detect/pcb_runs/expB_cbam_efficient/weights/best.pt"
OUT_NAME = "expB_cbam_efficient_val"

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    model.val(
        data    = "DSPCBSD+-1/data.yaml",
        imgsz   = 640,
        batch   = 8,
        device  = "0",
        plots   = True,
        project = "runs/detect/pcb_runs",
        name    = OUT_NAME,
        split   = "val",
    )
    print(f"\nPlots saved to: runs/detect/pcb_runs/{OUT_NAME}/")
