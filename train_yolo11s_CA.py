from ultralytics import YOLO
import torch
from pathlib import Path

if __name__ == "__main__":
    print("YOLO11s + Coordinate Attention training started")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Result folder
    result_root = Path("yolo11s_custom_results")
    result_root.mkdir(exist_ok=True)

    # Load custom YOLO11s + CA architecture
    model = YOLO("custom_models/yolo11s_ca.yaml")

    # Load YOLO11s pretrained weights where shapes match
    # New CoordAtt layers will train from scratch.
    model.load("yolo11s.pt")

    results = model.train(
        data="DSPCBSD+-1/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,              # safer for GTX 1650 with custom model
        device=0,
        project=str(result_root),
        name="yolo11s_ca_run1",
        workers=2,
        patience=10,
        amp=False,
        cache=False,
        plots=True,

        # Mild augmentation
        degrees=5,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        close_mosaic=10,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        erasing=0.2,
    )

    print("Training finished")