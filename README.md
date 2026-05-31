# PCB Surface Defect Detection with YOLOv11 + Attention Modules

---

## Overview

Printed circuit board (PCB) defects such as micro-scratches, hairline opens, and short circuits are extremely difficult to detect due to their small size, high visual similarity, and severe class imbalance. This project investigates whether adding attention mechanisms (CBAM / Coordinate Attention) to YOLOv11s improves detection performance, and compares against an unsupervised anomaly detection baseline (PatchCore) and inference-time enhancements (SAHI, TTA, ensemble/WBF).

---

## Dataset

**Deep Surface PCB Defect Dataset (DSPCBSD+)**
- Source: Roboflow (version 1, exported 2024-11-26)
- Total images: 7,367
- Classes: 9 defect classes (labelled `0`–`8` in the Roboflow export)
- Splits: Train / Val / Test (as defined in `data.yaml`)

---

## Hardware

All experiments ran on a consumer laptop GPU with tight VRAM constraints:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce GTX 1650 Ti — **4 GB VRAM** |
| CPU | Intel Core i7-10750H @ 2.60 GHz (6 cores) |
| RAM | 16 GB |

**VRAM rules (hard limits):**
- `YOLOv11s + imgsz=640 + batch=8 + AMP` → ✅ ~3.6 GB
- `YOLOv11s + CBAM/CA + imgsz=640 + batch=8 + AMP` → ✅ fits
- `YOLOv11s + P2 head + batch=8` → ❌ OOM
- `YOLOv11s + P2 head + batch=6 + AMP` → borderline / unstable
- `imgsz > 640`, `multi_scale`, `YOLOv11m` → ❌ OOM
- **AMP must always be enabled**

Approximate training speed: ~14.2 min/epoch (YOLOv11s, batch=8).

---

## Results Summary

All metrics are on the **validation set** unless stated otherwise.

### Detection Experiments

| Experiment | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|------------|--------|---------|--------------|-----------|--------|-------|
| Exp 0 — Baseline | 50 | 0.8388 | 0.4993 | 0.8087 | 0.7900 | Reference point |
| Exp 1 — Augmentation probe | 50 | 0.8287 | 0.4894 | 0.8019 | 0.7744 | Confounded — see below |
| **Exp A — Fixed Baseline** | **100** | **0.8244** | **0.4864** | **0.8192** | **0.7794** | **Primary baseline** |
| Exp 2a — Oversample | 28 | 0.8022 | 0.4454 | 0.8131 | 0.7661 | Not continued |
| ExpB — CBAM direct (50ep) | 50 | 0.7664 | 0.4292 | 0.7762 | 0.6697 | Underperformed |
| **ExpB — CA initial run (50ep)** | **50** | **0.8087** | **0.4537** | **0.7882** | **0.7572** | **Best attention run** |
| ExpB — CA freeze→unfreeze | 10+10 | 0.6580 | 0.3482 | 0.6744 | 0.6386 | Not continued |
| P2 head (best attempt) | 39 | 0.7338 | 0.4025 | 0.7444 | 0.7481 | Not continued — VRAM instability |

### Post-Processing & Inference Enhancements (on Exp A weights, test set)

| Method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|--------|-----------|--------|----|---------|--------------|
| Exp A baseline | 0.7739 | 0.7917 | 0.7827 | — | — |
| Exp A + threshold tuning | 0.8257 | 0.7585 | 0.7907 | — | — |
| Exp A + SAHI (slice=320, overlap=0.2) | 0.8317 | 0.7408 | 0.7836 | — | — |

### Ensemble / TTA Results (test set)

| Config | mAP@0.5 | mAP@0.5:0.95 | mAR@100 |
|--------|---------|--------------|---------|
| Exp A | 0.8152 | 0.4826 | 0.6124 |
| **Exp A + TTA** | **0.8211** | **0.4954** | **0.6376** |
| Exp B (CA) | 0.7495 | 0.4223 | 0.5882 |
| Exp B + TTA | 0.7621 | 0.4385 | 0.6173 |
| WBF ensemble (A + B) | 0.8050 | 0.4668 | 0.6070 |
| WBF + TTA | 0.7931 | 0.4636 | 0.6164 |

**Best overall: Exp A + TTA** (mAP@0.5 = 0.8211)

### Anomaly Detection Baseline (test set)

| Method | Precision | Recall | F1 | Image Detection Rate |
|--------|-----------|--------|----|----------------------|
| PatchCore (ResNet-18) | 0.1424 | 0.0828 | 0.1047 | 0.1555 |

PatchCore is unsuitable for this task — it cannot distinguish between defect classes and fails severely on a 9-class multi-label detection problem.

---

## Experiment Details

### Exp 0 — Baseline (50 epochs)

```
model=yolo11s.pt, epochs=50, batch=8, imgsz=640, amp=True
cls=0.5, erasing=0.4, auto_augment=randaugment, close_mosaic=10
```

Reference starting point. Not the primary baseline in the paper.

---

### Exp 1 — Augmentation Probe (50 epochs) — *confounded*

```
model=yolo11s.pt, epochs=50, batch=8, imgsz=640, amp=True
cls=1.0, erasing=0.0, auto_augment=randaugment, close_mosaic=10
```

**Why it failed — two variables changed simultaneously:**

1. **`cls=1.0` backfired.** Val cls_loss jumped 122% (0.862 → 1.916) with no accuracy gain. Doubling classification loss weight over-penalises classification relative to localisation.
2. **Close-mosaic spike at epoch 41.** Heavy augmentation made the model reliant on mosaic context; turning it off caused a hard loss spike that never recovered.

Reported in the paper as a preliminary investigation only. Both issues informed the Exp A design.

---

### Exp A — Fixed Strong Baseline (100 epochs) — *primary baseline*

```
model=yolo11s.pt, epochs=100, batch=8, imgsz=640, amp=True
cls=0.5, erasing=0.4, auto_augment=randaugment, close_mosaic=0, cache=True
```

Reverts cls to 0.5, disables close_mosaic to avoid the epoch-41 spike, runs 100 epochs. This is the paper's primary baseline. Best single-model result: **mAP@0.5 = 0.8244**.

---

### Exp 2a — Oversampling (not continued)

Attempted class-balance correction via 2.67× oversampling of weak classes. Stopped at epoch 28 — did not improve over Exp A and introduced training instability.

---

### Exp B — CBAM (direct, 50 epochs)

```
model=yolo11s_cbam.yaml, epochs=50, batch=8, imgsz=640, amp=True
```

CBAM inserted directly into the YOLOv11s backbone, trained from pretrained weights. Underperformed Exp A significantly (mAP@0.5 = 0.7664). Likely insufficient epochs to recover attention head convergence.

---

### Exp B — Coordinate Attention (CA), initial run 50 epochs — *best attention result*

```
model=yolo11s_ca.yaml, epochs=50, batch=8, imgsz=640, amp=True
```

CA module integrated into the backbone. Best attention-based result: **mAP@0.5 = 0.8087**, still below Exp A. The gap suggests attention modules add complexity without proportional benefit at this scale and VRAM budget.

---

### Exp B — CA freeze → unfreeze (not continued)

Two-phase training: freeze backbone for 10 epochs, then unfreeze for fine-tuning. Stopped after phase 2 — mAP@0.5 = 0.6580, training was unstable and not progressing.

---

### P2 Head Experiments (all discontinued)

Multiple attempts to add a stride-4 P2 detection head for small-object detection. All failed or were not continued due to VRAM instability, loss explosion, or convergence failure at batch sizes that fit in 4 GB. The best attempt reached mAP@0.5 = 0.7338 at epoch 39 before being stopped. **SAHI was adopted as a zero-training alternative** for small-object inference.

---

### Exp C — PatchCore (ResNet-18)

Unsupervised anomaly detection baseline. Memory bank: 208,700 patches (5% coreset), k-NN k=5. Precision = 0.1424, F1 = 0.1047. Cannot classify defect type; designed for binary anomaly detection, not 9-class multi-label object detection. Included in the paper as a comparison point to motivate supervised approaches.

---

### SAHI — Slicing Aided Hyper Inference

Applied to Exp A best weights. Two configurations tested:

- **slice=320, overlap=0.2, tuned thresholds:** P=0.8317, R=0.7408, F1=0.7836
- **slice=640, overlap=0.2, conf=0.25:** overall mAP@0.5=0.5308 (lower conf threshold changes trade-off)

SAHI marginally improved precision but at the cost of recall. Net F1 gain was minimal (+0.008). Not the primary improvement vector.

---

### Threshold Tuning

Per-class confidence thresholds optimised on the validation set:

| Class | Threshold |
|-------|-----------|
| 0 | 0.45 |
| 1 | 0.35 |
| 2 | 0.50 |
| 3 | 0.35 |
| 4 | 0.35 |
| 5 | 0.70 |
| 6 | 0.30 |
| 7 | 0.25 |
| 8 | 0.40 |

Improved test precision from 0.7739 → 0.8257 and F1 from 0.7827 → 0.7907.

---

### Exp E — Ensemble + WBF + TTA

WBF (Weighted Box Fusion) ensemble of Exp A and Exp B predictions, with and without Test-Time Augmentation. Best configuration: **Exp A + TTA** (mAP@0.5 = 0.8211). WBF ensemble did not outperform single-model TTA, suggesting the weaker Exp B predictions hurt rather than helped the fusion.

---

## Key Findings

1. **The fixed baseline (Exp A) is the strongest single model.** 100 epochs with corrected hyperparameters beat all attention-augmented variants.
2. **CA outperformed CBAM** in the attention experiments (0.8087 vs 0.7664 mAP@0.5), but neither exceeded the baseline.
3. **P2 head is not viable on 4 GB VRAM.** All P2 attempts were discontinued. SAHI achieves similar small-object benefit at inference with no VRAM cost.
4. **PatchCore is not suitable** for multi-class PCB defect detection — it is a binary anomaly detector by design.
5. **TTA provides the clearest improvement** over standard inference (+0.006 mAP@0.5 on test set) with zero retraining cost.

---

## Repository Structure

```
PCP Defection/
├── Experiments_Results_Compined/
│   ├── 1.Intial_Runs/
│   │   ├── exp0_baseline/
│   │   └── exp1_augmentation/
│   ├── 2.Based_on_Fixed_Baseline/
│   │   ├── expA_fixed_baseline_100ep/
│   │   ├── exp2a_oversample_267/
│   │   ├── best_100ep_sahi_full_metrics/
│   │   ├── sahi_results.csv
│   │   ├── sahi_summary.csv
│   │   ├── threshold_tuning_results.csv
│   │   └── threshold_tuning_summary.csv
│   ├── 3.CBAM_Runs/
│   │   ├── ExpB_cbam_Direct_on_data_50ep/
│   │   ├── [BEST]_ExpB_yolo11s_CA_intial_run_50ep/
│   │   ├── ExpB_CA_from_100ep_Freeze,unfreeze/
│   │   └── [not_contd]_ExpB_cbam_on_100_/
│   ├── 4.P2_Runs/                          # All discontinued
│   └── 5.Else/
│       ├── expC_patchcore/
│       └── expE_ensemble/
├── PCB_Paper_IEEE.docx                     # Main IEEE paper draft
├── PCB_Paper_Full.docx                     # Extended version
├── PCB_Experimental_Summary.docx           # Experiment log
├── Project plan.docx                       # Execution plan + paper outline
└── README.md

DSPCBSD/                                    # (separate mounted folder)
├── DSPCBSD+-1/                             # Dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── experiments/                            # Training scripts
│   ├── Train_Exp0_baseline.py
│   ├── Train_Exp1_augmentation.py
│   ├── Train_Exp_A_FixedBaseline.py
│   ├── Train_Exp_B_CBAM.py
│   ├── ExpC_PatchCore.py
│   ├── ExpD_SAHI.py
│   ├── expE_ensemble_wbf.py
│   ├── ExpF_ThresholdTuning.py
│   └── compare_experiments.py
├── yolo11s.pt                              # Pretrained weights
└── yolo11s_cbam.yaml                       # Custom model config
```

---

## Dependencies

```
ultralytics>=8.3
torch>=2.0
torchvision
sahi
scikit-learn
matplotlib
```

---

