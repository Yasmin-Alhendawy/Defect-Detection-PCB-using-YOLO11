from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("Script started")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11s.pt")
    print("Model loaded")

    results = model.train(
        data="DSPCBSD+-1/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project="pcb_runs",
        name="yolo11s_run1",
        patience=10,
        workers=4,
        amp=False
    )

    print("Training finished")