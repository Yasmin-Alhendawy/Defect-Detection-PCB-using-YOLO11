# Defect Detection PCB using YOLO

This repository contains code for PCB defect detection using YOLO-based models.

## Overview
- Project: Defect Detection for PCB images using a YOLO implementation.

## Quick setup
1. Create (already done) and activate the virtual environment:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - Command Prompt: `.\.venv\Scripts\activate.bat`
   - Git Bash / WSL: `source .venv/bin/activate`
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Prepare your dataset and configuration (place dataset in a `data/` folder or update training scripts accordingly).

## Usage
- Training / inference commands depend on the YOLO implementation in this repo. Typical commands:

```powershell
# train
python train.py --config config.yaml

# inference
python detect.py --weights runs/exp/weights/best.pt --source data/images
```

Replace the scripts and arguments above with the actual entry points in this repo.

## Git / GitHub
- Add `.venv/` to `.gitignore` to avoid committing the virtual environment.
- To commit and push this README locally:

```powershell
git add README.md
git commit -m "Add README"
git remote add origin https://github.com/Yasmin-Alhendawy/Defect-Detection-PCB-using-YOLO11.git
git push -u origin main
```

If your default branch is `master` or another name, replace `main` accordingly. For SSH remotes use the SSH URL instead of HTTPS.

## Notes
- If you want, I can commit and push this README for you — tell me whether you prefer SSH or HTTPS+PAT, or provide credentials.
