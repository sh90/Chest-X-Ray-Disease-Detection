
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

# About the Project:
### We will build a small but real-world ML project: 
A model that flags possible pneumonia on chest X-rays. It’s not a medical device and it doesn’t replace radiologists. Think of it as a triage assistant that can surface the scariest cases first and help us reason about why the model thinks so.”

“We built a binary classifier (NORMAL vs PNEUMONIA) using transfer learning on ResNet-18.”

“Images are preprocessed to 224×224, normalized like ImageNet, and we use light augmentation (crop/flip/rotate) for generalization.”

“Training happens in two phases:

Freeze the pretrained backbone and train only the classifier head;

Unfreeze and fine-tune the whole network with a smaller learning rate.”

“We evaluate with accuracy, Precision Recall, and we tune the decision threshold for the best F1.”

“For interpretability, we generate Grad-CAM heatmaps to see where the model looked when it made a call.”

“There’s a Streamlit UI with a step-by-step wizard: Data → Preview → Train → Evaluate → Interpret, plus single-image analysis and an error gallery.


