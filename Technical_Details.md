## What the blocks are

The model’s last conv layer (ResNet-18 layer4) is only 7×7 cells.

Grad-CAM is computed on that 7×7 map and then stretched back to the image → you see blocks.

Color = positive evidence for the class you’re explaining (usually the predicted class).
Red/orange = strong positive contribution, green = weak, blue = none (negative evidence is clipped by ReLU).

## How to read one overlay (quick script)

State the labels: “Predicted X, True Y.”

Locate the hot area(s): “Where is the red? Upper left lung? Outside the lungs?”

Judge plausibility: “Inside lung fields → plausible. On text corners/diaphragm/borders → likely a shortcut.”

Key caveat: “This is not a segmentation of disease—just where the network looked to support its decision.”

## Why they’re blocky (say this once)

“ResNet compresses a 224×224 image down to 7×7 features before the classifier.
Each square summarizes a big patch of the original X-ray (roughly ~32×32 pixels). That’s why it looks chunky.”

Using the Error Gallery (teaching beats)

Most confident wrong: “These are the model’s strongest illusions.” Ask: “What non-lung cues might it be using?”

Random: Compare with a few correct cases: “When correct, attention tends to sit within lung fields;
when wrong, it drifts to borders/labels/devices.”

Red ≠ pneumonia mask; it’s evidence for a class.

## Attention should be largely inside lungs.

Hot spots on text markers, edges, or medical devices = spurious correlation.

Confidence ≠ correctness—high-confidence wrong is where shortcuts show.

If you want less blockiness (optional upgrade)

Use an earlier layer for Grad-CAM:

layer3[-1] → ~14×14 (smoother)

layer2[-1] → ~28×28 (smoothest, but less semantic)

Or upsample the CAM to input size before overlay (we can toggle this in code if you want).

## Common pitfalls to call out

Outside-lung focus: crop or mask borders; remove text markers.

Devices (tubes/lines) drive predictions: data bias; consider filtering or labeling such cases.

Tiny focal opacities: Grad-CAM may miss small lesions due to coarse resolution—don’t over-interpret.


“This heatmap shows where the model found positive evidence for its chosen class.
Because we compute it on a 7×7 feature map, it looks blocky.
Red means ‘this region pushed the prediction up,’ blue means ‘it didn’t help.’ It’s not a segmentation mask.
In this misclassified example, notice the hot area near the corner—outside the lungs—suggesting the model
latched onto a shortcut (like text or border). In contrast, here’s a correct case: the hot blocks sit over lung fields,
which is more clinically plausible.
Grad-CAM helps us debug what the model uses, but we still need careful datasets and evaluation.”
