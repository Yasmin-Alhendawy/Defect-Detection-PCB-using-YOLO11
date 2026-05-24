# Project Context: PCB Defect Detection Ablation Study

## Project Root
C:\Users\Yasmi\OneDrive\Documents\DSPCBSD

## Structure
- Dataset folder: DSPCBSD+-1/
- Dataset config: DSPCBSD+-1/data.yaml
- Train images: DSPCBSD+-1/train/images/
- Train labels: DSPCBSD+-1/train/labels/
- Pretrained weights: yolo11s.pt
- Virtual env: .venv

## Experiment 0 (DONE — DO NOT TOUCH)
- Script: train.py
- Model: yolo11s.pt
- epochs=50, imgsz=640, batch=8, device=0
- Results: runs/detect/pcb_runs/yolo11s_run1-1/

## Experiment 1 (READY TO RUN)
- Script: experiments/exp1_augmentation/train_exp1.py
- Same model and settings as Exp 0
- Results will save to: runs/detect/pcb_runs/exp1_augmentation
- Three changes: Albumentations, Copy-Paste, Focal Loss fl_gamma=2.0