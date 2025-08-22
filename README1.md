
# Chest X-Ray Disease Detection (Modular, Python 3.12)

Modular teaching project: train a binary classifier (NORMAL vs PNEUMONIA) on chest X-rays, evaluate with Precision/Recall, and visualize Grad-CAM heatmaps. Includes Streamlit tools and a step-by-step CLI.

## Download Dataset
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

## Structure
```
cxr_demo_modular/
  README.md
  requirements.txt
  config.py
  utils.py
  data.py
  modeling.py
  training.py
  evaluation.py
  gradcam.py
  steps_cli.py
  streamlit_new.py
```
## Setup
```
pip install -r requirements.txt
```

## Run (end-to-end)
```
python steps_cli.py
streamlit run streamlit_new.py
```

## Streamlit UI
```bash
streamlit run streamlit_app.py
```
Modes:
- Step-by-step Wizard (Data → Preview → Train → Evaluate → Interpret)
- Single Image (upload → probs + Grad-CAM)
- Error Gallery (scan test misclassifications)
- Threshold Tuning (optimize F1)
- Export Grad-CAMs (ZIP)

## CLI (run steps via PyCharm/terminal)
```bash
python scripts/steps_cli.py prep --data_dir data/chest_xray
python scripts/steps_cli.py dataloader --data_dir data/chest_xray --out_dir outputs
python scripts/steps_cli.py train --data_dir data/chest_xray --epochs 2 --freeze_epochs 1 --out_dir outputs --save_model
python scripts/steps_cli.py eval --data_dir data/chest_xray --weights outputs/cxr_resnet18.pt --out_dir outputs
python scripts/steps_cli.py cam --data_dir data/chest_xray --weights outputs/cxr_resnet18.pt --out_dir outputs --num 8
python scripts/steps_cli.py tune --data_dir data/chest_xray --weights outputs/cxr_resnet18.pt --out_dir outputs
```

## Dataset (Kaggle)
```bash
mkdir -p data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```
Point `--data_dir` to `data/chest_xray`.
