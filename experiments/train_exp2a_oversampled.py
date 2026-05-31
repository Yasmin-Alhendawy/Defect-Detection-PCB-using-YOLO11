"""
exp2a: Warm-start fine-tune from expA on the OVERSAMPLED dataset (classes 2/6/7 boosted).
Backbone frozen, mild augmentation, ~40 epochs. Designed to fit GTX 1650 Ti (4GB).

NOTE: Windows requires the `if __name__ == "__main__":` guard for multiprocessing
dataloader workers, otherwise child processes re-import this file infinitely.
"""

from ultralytics import YOLO


def main():
    model = YOLO(
        r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\runs\detect\pcb_runs\expA_fixed_baseline\weights\best.pt"
    )

    results = model.train(
        data=r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\DSPCBSD+-1\data_os.yaml",
        project=r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\runs\detect\pcb_runs",
        name="exp2a_oversample_267",

        # core schedule
        epochs=40,
        patience=15,
        batch=4,            # reduced from 6 — AMP gets disabled on 1650 Ti, doubles VRAM
        imgsz=640,
        amp=True,           # Ultralytics will still disable; harmless to keep

        # warm-start: freeze backbone, train neck+head only -> faster, fits VRAM
        freeze=10,

        # optimizer
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.0,

        # SLIGHT augmentation only — duplicates already provide the boost
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,

        # housekeeping
        save=True,
        save_period=5,
        workers=2,
        cache=False,
        verbose=True,
        plots=True,
        seed=42,
    )
    print("Done. Best weights:", results.save_dir)  # type: ignore

    # ========================================================================
    # Explicit validation pass — full plots + confusion matrix + per-class table
    # ========================================================================
    print("\n" + "=" * 70)
    print("Running final validation with full plots...")
    print("=" * 70)

    best_weights = str(results.save_dir) + r"\weights\best.pt"  # type: ignore
    val_model = YOLO(best_weights)

    val_results = val_model.val(
        data=r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\DSPCBSD+-1\data.yaml",  # ORIGINAL val set
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        save_json=True,
        project=r"C:\Users\Yasmi\OneDrive\Documents\DSPCBSD\runs\detect\pcb_runs",
        name="exp2a_oversample_267_val",
        split="val",
        verbose=True,
    )

    # Per-class metrics to terminal
    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS (validation set)")
    print("=" * 70)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    print(f"{'Class':<8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
    print("-" * 50)
    p, r, ap50, ap = val_results.box.p, val_results.box.r, val_results.box.ap50, val_results.box.ap
    for i, name in enumerate(class_names):
        print(f"{name:<8}{p[i]:>10.4f}{r[i]:>10.4f}{ap50[i]:>10.4f}{ap[i]:>12.4f}")
    print("-" * 50)
    print(f"{'MEAN':<8}{p.mean():>10.4f}{r.mean():>10.4f}{ap50.mean():>10.4f}{ap.mean():>12.4f}")
    print("\nConfusion matrix saved to:")
    print(r"  runs\detect\pcb_runs\exp2a_oversample_267_val\confusion_matrix_normalized.png")


if __name__ == "__main__":
    main()
