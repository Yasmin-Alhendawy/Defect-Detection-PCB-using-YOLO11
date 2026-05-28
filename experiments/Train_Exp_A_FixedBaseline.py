from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")

    model.train(
        # --- Core ---
        data="DSPCBSD+-1/data.yaml",
        epochs=100,
        batch=8,
        imgsz=640,
        device="0",

        # --- Naming ---
        project="pcb_runs",
        name="expA_fixed_baseline",
        exist_ok=False,

        # --- P1 Fixes ---
        cls=0.5,
        close_mosaic=20,
        erasing=0.0,

        # --- LR for 100 epochs ---
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        patience=50,

        # --- Keep from exp0 ---
        optimizer="auto",
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        momentum=0.937,
        box=7.5,
        dfl=1.5,
        auto_augment="randaugment",
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,

        # --- Infrastructure ---
        amp=True,
        workers=4,
        seed=0,
        deterministic=True,
        pretrained=True,
        plots=True,
        save=True,
    )