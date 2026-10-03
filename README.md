# 🔬 OncoScan — Early-Stage Cancer Cell Detection

Transfer-learned CNN (ResNet-18, PyTorch) that flags malignant cells in histopathology
images, tuned for **high recall** in a screening setting (missed cancers cost more than false alarms).

**Techniques:** augmentation (flips, rotations, stain-style colour jitter) · class-imbalance handling
(minority oversampling + weighted loss) · recall-targeted threshold selection · Grad-CAM explainability · Gradio upload UI.

> Research/portfolio demo only — not a medical device.

## Quick start (Mac, Apple Silicon or Intel)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The device is auto-selected: Apple GPU (MPS) → CUDA → CPU.

### 1. Smoke test (no downloads, ~1 min)
```bash
python -m oncoscan.train --dataset synthetic --epochs 3 --size 64 --max-samples 300
python app.py          # open the printed http://127.0.0.1:7860 URL and upload an image
```
Synthetic data only proves the pipeline works — it is not clinically meaningful.

### 2. Real data
**PatchCamelyon (auto-download, ~7 GB for full set — start with a subset):**
```bash
python -m oncoscan.train --dataset pcam --data-dir data --max-samples 20000 --epochs 5 --size 96
```
**BreakHis (manual download from the official site)** — arrange as:
```
data/breakhis/benign/...png
data/breakhis/malignant/...png
```
```bash
python -m oncoscan.train --dataset folder --data-dir data/breakhis --size 128 --epochs 8
```
Use `--target-recall 0.97` to trade precision for fewer missed cancers.
Outputs: `runs/oncoscan.pt` (weights + tuned threshold) and `runs/oncoscan.json` (metrics).

### 3. Run the app
```bash
python app.py --ckpt runs/oncoscan.pt
```
Shows malignant probability, a screening verdict at the recall-tuned threshold, and a Grad-CAM heatmap.

## Tests
`pytest -q`

## Project layout
`oncoscan/data.py` datasets/augmentation/imbalance · `model.py` ResNet-18 · `train.py` training + threshold tuning ·
`metrics.py` recall-first threshold · `gradcam.py` · `predict.py` · `app.py` Gradio UI.

## Notes for the write-up
Report recall/precision/PR-AUC on a held-out *patient-level* split (the code splits by image;
for real data split by patient to avoid leakage). Early-stage positives are rare, so evaluate recall on that subgroup separately.
