
# streamlit_app.py — dtype-stable + robust loader + backward-compatible UI
import io, os, random, zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageOps

import torch
import torch.nn.functional as F
import streamlit as st

from config import IMG_SIZE, NORM_MEAN, NORM_STD
from modeling import build_model
from gradcam import GradCAM, overlay_cam_on_image

# Optional imports for dataset-based modes
try:
    from data import build_transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
except Exception:
    build_transforms = None
    datasets = None
    DataLoader = None

# ----------------------
# Global settings & dtype
# ----------------------
TORCH_DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Chest X-Ray Teaching Tools", layout="wide")
st.title("🩺 Chest X-Ray Disease Detection — Teaching Tools")
st.caption("Educational demo — not for clinical use.")

# ---------
# UI helpers
# ---------
def safe_st_image(img, caption=None):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

def safe_pyplot(fig, container=None):
    target = container or st
    try:
        target.pyplot(fig, use_container_width=True)
    except TypeError:
        target.pyplot(fig)

# -----------------
# Robust file loader
# -----------------
def load_uploaded_image(up):
    """Return a PIL.Image in RGB (handles PNG/JPG; optional DICOM if pydicom available)."""
    data = up.read()
    up.seek(0)  # rewind for Streamlit
    name = (up.name or "").lower()

    # DICOM (optional)
    if name.endswith(".dcm"):
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.ptp() + 1e-6)
            arr = (arr * 255).clip(0, 255).astype("uint8")
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                img = Image.fromarray(arr[..., :3]).convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM: {e}")

    # PNG/JPG
    try:
        img = Image.open(io.BytesIO(data))
        img.load()                         # force decode now
        img = ImageOps.exif_transpose(img) # correct orientation
        img = img.convert("RGB")
        return img
    except UnidentifiedImageError:
        raise RuntimeError("Not a valid image file (expected PNG/JPG or DICOM).")

# ---------------
# Preprocess image
# ---------------
def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("L").convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std  = np.array(NORM_STD,  dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # CHW
    t = torch.from_numpy(arr).unsqueeze(0)  # [1,3,H,W]
    return t.to(dtype=TORCH_DTYPE)

def plot_probs(probs, labels):
    fig = plt.figure(figsize=(4, 3))
    xs = np.arange(len(labels))
    plt.bar(xs, probs)
    plt.xticks(xs, labels, rotation=15)
    plt.ylim(0, 1)
    plt.title("Predicted Probabilities")
    plt.tight_layout()
    return fig

def scan_split(split_dir, batch_size=32):
    """Return (paths, y_true, probs_all, y_pred)."""
    if datasets is None or build_transforms is None:
        st.error("Missing torchvision / data utilities.")
        return None
    tfm = build_transforms(train=False)
    ds = datasets.ImageFolder(split_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    paths = [p for (p, _) in ds.samples]
    y_true, probs_all, y_pred = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=TORCH_DTYPE)
            pr = torch.softmax(model(x), dim=1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            probs_all.extend(pr.tolist())
            y_pred.extend(np.argmax(pr, axis=1).tolist())
    return paths, np.array(y_true), np.array(probs_all), np.array(y_pred)

# ---------------
# Sidebar controls
# ---------------
with st.sidebar:
    st.header("Global Settings")
    mode = st.radio(
        "Mode",
        options=["Step-by-step Wizard", "Single Image", "Error Gallery", "Threshold Tuning", "Export Grad-CAMs"],
        index=0,
    )
    data_dir = st.text_input("Dataset folder", value="", help="Must contain train/val/test")
    default_classes = ["NORMAL", "PNEUMONIA"]
    auto_classes = None
    if data_dir:
        tr = os.path.join(data_dir, "train")
        try:
            auto_classes = sorted([d for d in os.listdir(tr) if os.path.isdir(os.path.join(tr, d))])
        except Exception:
            auto_classes = None
    class_names = auto_classes if auto_classes else default_classes
    st.write(f"Classes: {class_names}")

    weights_path = st.text_input("Weights path", value="outputs/cxr_resnet18.pt")
    up_weights = st.file_uploader("...or upload .pt", type=["pt", "pth"])
    threshold = st.slider("Default decision threshold", 0.0, 1.0, 0.5, 0.01)

# ------------------------
# Build model (float32)
# ------------------------
model = build_model(num_classes=len(class_names), pretrained=True).to(device).to(TORCH_DTYPE)

# Load weights (downcast if needed)
loaded = False
state = None
if up_weights is not None:
    try:
        state = torch.load(io.BytesIO(up_weights.read()), map_location="cpu")
        loaded = True
    except Exception as e:
        st.error(f"Failed uploaded weights: {e}")
elif weights_path and os.path.exists(weights_path):
    try:
        state = torch.load(weights_path, map_location="cpu")
        loaded = True
    except Exception as e:
        st.warning(f"Could not load weights from path: {e}")

if loaded and state is not None:
    try:
        # Force any float64 tensors to float32 before loading
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                state[k] = v.float()
        model.load_state_dict(state)
        model = model.to(TORCH_DTYPE)
        st.success("Model weights loaded.")
    except Exception as e:
        st.error(f"Error applying state dict: {e}")
else:
    st.info("Using randomly initialized weights (for demo only).")

# =====================
# Mode: Step-by-step Wizard
# =====================
if mode == "Step-by-step Wizard":
    st.subheader("Walk through the pipeline step by step")

    if "wiz" not in st.session_state:
        st.session_state.wiz = 1
    steps = ["Data Check", "Preview Batch", "Train", "Evaluate", "Interpret"]
    cols = st.columns(5)
    for i, label in enumerate(steps, start=1):
        cols[i-1].button(f"{i}. {label}", on_click=lambda s=i: st.session_state.update(wiz=s))

    step = st.session_state.wiz
    st.write(f"### Step {step}: {steps[step-1]}")

    if step == 1:
        if not data_dir:
            st.info("Enter dataset path in sidebar.")
        else:
            try:
                from torchvision import datasets as _ds
                ds_tr = _ds.ImageFolder(os.path.join(data_dir, "train"))
                ds_va = _ds.ImageFolder(os.path.join(data_dir, "val"))
                ds_te = _ds.ImageFolder(os.path.join(data_dir, "test"))
                st.write(f"Classes: {ds_tr.classes}")
                def counts(ds):
                    if hasattr(ds, "samples"):
                        ys = [y for _, y in ds.samples]
                    elif hasattr(ds, "targets"):
                        ys = ds.targets
                    else:
                        ys = [y for _, y in ds.imgs]
                    out = {c: 0 for c in ds.classes}
                    for y in ys:
                        out[ds.classes[y]] += 1
                    return out
                st.write("Train:", counts(ds_tr))
                st.write("Val:", counts(ds_va))
                st.write("Test:", counts(ds_te))
                st.success("Click Next to preview a batch.")
            except Exception as e:
                st.error(f"Data check failed: {e}")

    elif step == 2:
        if not data_dir:
            st.error("Set dataset path first.")
        else:
            try:
                from data import build_datasets, build_dataloaders
                data = build_datasets(data_dir)
                loaders = build_dataloaders(data, batch_size=8, num_workers=0)
                imgs, labels = next(iter(loaders["train"]))
                n = min(8, imgs.size(0))
                cols_n = min(4, n)
                rows = int(np.ceil(n / cols_n))
                fig = plt.figure(figsize=(3 * cols_n, 3 * rows))
                inv = {v: k for k, v in data["train"].class_to_idx.items()}
                for i in range(n):
                    ax = fig.add_subplot(rows, cols_n, i + 1)
                    im = imgs[i].permute(1, 2, 0).numpy()
                    std = np.array([0.229, 0.224, 0.225])
                    mean = np.array([0.485, 0.456, 0.406])
                    ax.imshow(np.clip(std * im + mean, 0, 1))
                    ax.set_title(inv[int(labels[i])])
                    ax.axis("off")
                safe_pyplot(fig)
                st.info("Talk through transforms & normalization.")
            except Exception as e:
                st.error(f"Preview failed: {e}")

    elif step == 3:
        e1, e2, bs, lr = st.columns(4)
        epochs = e1.number_input("Epochs", 1, 50, 2)
        freeze = e2.number_input("Freeze epochs", 0, 10, 1)
        batch = bs.selectbox("Batch size", [8, 16, 32], 1)
        lr = lr.number_input("LR", 1e-5, 1e-1, 1e-3, format="%.1e")
        go = st.button("Start Training")
        if go:
            from data import build_datasets, build_dataloaders
            from training import train_one_epoch, evaluate
            from utils import device_auto, seed_everything
            from modeling import build_model

            seed_everything(42)
            dev = device
            data = build_datasets(data_dir)
            loaders = build_dataloaders(data, batch_size=batch, num_workers=0)
            cls = list(data['train'].class_to_idx.keys())
            mdl = build_model(num_classes=len(cls), pretrained=True).to(dev).to(TORCH_DTYPE)

            def tog(flag):
                for m in [mdl.conv1, mdl.bn1, mdl.layer1, mdl.layer2, mdl.layer3, mdl.layer4]:
                    for p in m.parameters():
                        p.requires_grad = flag

            hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            prog = st.progress(0)
            opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)

            for ep in range(freeze):
                tr, ta = train_one_epoch(mdl, loaders['train'], opt, dev)
                vl, va, _, _ = evaluate(mdl, loaders['val'], dev)
                hist['train_loss'].append(tr); hist['val_loss'].append(vl)
                hist['train_acc'].append(ta);   hist['val_acc'].append(va)
                prog.progress(int(100 * ((ep + 1) / max(1, epochs))))

            tog(True)
            opt = torch.optim.AdamW(mdl.parameters(), lr=lr * 0.5)
            for i in range(max(0, epochs - freeze)):
                ep = freeze + i + 1
                tr, ta = train_one_epoch(mdl, loaders['train'], opt, dev)
                vl, va, _, _ = evaluate(mdl, loaders['val'], dev)
                hist['train_loss'].append(tr); hist['val_loss'].append(vl)
                hist['train_acc'].append(ta);   hist['val_acc'].append(va)
                prog.progress(int(100 * (ep / max(1, epochs))))

            st.session_state.w_model = {k: v.cpu().to(TORCH_DTYPE) for k, v in mdl.state_dict().items()}
            st.session_state.w_classes = cls

            from training import plot_training
            plot_training(hist)  # saved/embedded; no pop-ups
            st.success("Training finished. Click Evaluate.")

    elif step == 4:
        if "w_model" not in st.session_state:
            st.warning("Train first.")
        else:
            from data import build_datasets, build_dataloaders
            from modeling import build_model
            from sklearn.metrics import roc_auc_score
            from training import evaluate

            data = build_datasets(data_dir)
            loaders = build_dataloaders(data, batch_size=16, num_workers=0)
            cls = st.session_state.w_classes
            mdl = build_model(num_classes=len(cls), pretrained=False).to(device).to(TORCH_DTYPE)
            mdl.load_state_dict(st.session_state.w_model)
            tl, ta, y_true, y_score = evaluate(mdl, loaders['test'], device)
            try:
                auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc = float("nan")
            st.json({"test_loss": tl, "test_acc": ta, "roc_auc": auc})

            from evaluation import plot_confusion_and_curves
            plot_confusion_and_curves(y_true, y_score, cls)

    elif step == 5:
        if "w_model" not in st.session_state:
            st.warning("Train first.")
        else:
            from data import build_datasets, build_dataloaders
            from modeling import build_model

            data = build_datasets(data_dir)
            loaders = build_dataloaders(data, batch_size=8, num_workers=0)
            cls = st.session_state.w_classes
            mdl = build_model(num_classes=len(cls), pretrained=False).to(device).to(TORCH_DTYPE)
            mdl.load_state_dict(st.session_state.w_model)
            mdl.eval()
            target = mdl.layer4[-1]
            cam = GradCAM(mdl, target)

            for x, _ in loaders['test']:
                x = x.to(device=device, dtype=TORCH_DTYPE)
                with torch.no_grad():
                    pred = mdl(x).argmax(1)
                x.requires_grad_(True)
                with torch.enable_grad():
                    cam_map = cam(x, class_idx=pred).squeeze(1).cpu().numpy()
                n = min(x.size(0), 6)
                cols = st.columns(n)
                for i in range(n):
                    fig = overlay_cam_on_image(x[i].detach().cpu(), cam_map[i])
                    safe_pyplot(fig, container=cols[i])
                    plt.close(fig)
                break
            cam.close()

# =====================
# Mode: Single Image
# =====================
elif mode == "Single Image":
    st.subheader("Upload an image")
    up = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "dcm"])
    use_pred = st.checkbox("Explain predicted class", value=True)
    manual = st.selectbox("Manual class index", list(range(len(class_names))), index=min(1, len(class_names) - 1), disabled=use_pred)

    if up is not None:
        try:
            img = load_uploaded_image(up)
        except Exception as e:
            st.error(str(e))
            st.stop()

        safe_st_image(img, "Uploaded")

        x = preprocess(img).to(device=device, dtype=TORCH_DTYPE)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy().flatten()

        pred = int(np.argmax(probs))
        st.markdown(f"**Prediction:** {class_names[pred]} • **Prob:** {probs[pred]:.3f}")
        fig = plot_probs(probs, class_names)
        safe_pyplot(fig)

        # Grad-CAM
        target_layer = model.layer4[-1]
        cam = GradCAM(model, target_layer)
        target_idx = torch.tensor([pred if use_pred else manual], dtype=torch.long, device=device)
        x.requires_grad_(True)
        with torch.enable_grad():
            cam_map = cam(x, class_idx=target_idx).squeeze(0).squeeze(0).cpu().numpy()
        fig = overlay_cam_on_image(x[0].detach().cpu(), cam_map)
        safe_pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        st.download_button("Download Grad-CAM overlay (PNG)", data=buf.getvalue(), file_name="gradcam_overlay.png", mime="image/png")
        cam.close()

# =====================
# Mode: Error Gallery
# =====================
elif mode == "Error Gallery":
    st.subheader("Misclassified Test Images")
    if datasets is None:
        st.error("Missing torchvision/data utils.")
    else:
        cA, cB, cC = st.columns(3)
        with cA:
            max_items = st.slider("Max images", 4, 60, 16, 2)
        with cB:
            batch_size = st.selectbox("Batch size (scan)", [8, 16, 32, 64], 1)
        with cC:
            order = st.selectbox("Order", ["Most confident wrong", "Random"], 0)

        if st.button("Scan Test Split"):
            test_dir = os.path.join(data_dir, "test")
            if not (data_dir and os.path.isdir(test_dir)):
                st.error("Provide a valid dataset folder.")
            else:
                out = scan_split(test_dir, batch_size=batch_size)
                if out is None:
                    st.stop()
                paths, y_true, probs, y_pred = out
                conf = probs.max(axis=1)
                mis = np.where(y_true != y_pred)[0].tolist()
                if not mis:
                    st.success("No misclassifications found.")
                else:
                    if order == "Most confident wrong":
                        mis.sort(key=lambda i: float(conf[i]), reverse=True)
                    else:
                        random.shuffle(mis)
                    mis = mis[:max_items]

                    target_layer = model.layer4[-1]
                    cam = GradCAM(model, target_layer)
                    ncols = 4
                    rows = (len(mis) + ncols - 1) // ncols
                    for r in range(rows):
                        cols = st.columns(ncols, gap="small")
                        for c in range(ncols):
                            k = r * ncols + c
                            if k >= len(mis):
                                break
                            i = mis[k]
                            img = Image.open(paths[i]).convert("RGB")
                            x = preprocess(img).to(device=device, dtype=TORCH_DTYPE)
                            target_idx = torch.tensor([int(y_pred[i])], dtype=torch.long, device=device)
                            x.requires_grad_(True)
                            with torch.enable_grad():
                                cam_map = cam(x, class_idx=target_idx).squeeze(0).squeeze(0).cpu().numpy()
                            fig = overlay_cam_on_image(x[0].detach().cpu(), cam_map)
                            safe_pyplot(fig, container=cols[c])
                            plt.close(fig)
                            cols[c].caption(f"Pred: {class_names[int(y_pred[i])]} ({conf[i]:.2f}) | True: {class_names[int(y_true[i])]}")
                    cam.close()

# =====================
# Mode: Threshold Tuning
# =====================
elif mode == "Threshold Tuning":
    st.subheader("Optimize F1 (binary)")
    if len(class_names) != 2 or datasets is None:
        st.error("Binary task + torchvision required.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pos_idx = st.selectbox("Positive class index", [0, 1], 1, format_func=lambda i: f"{i} — {class_names[i]}")
        with c2:
            batch_size = st.selectbox("Batch size", [16, 32, 64], 1)
        with c3:
            run = st.button("Compute")

        if run:
            test_dir = os.path.join(data_dir, "test")
            out = scan_split(test_dir, batch_size=batch_size)
            if out is None:
                st.stop()
            _, y_true, probs, _ = out
            y_score = probs[:, pos_idx]
            ts = np.linspace(0, 1, 201)
            best = {"threshold": None, "f1": -1, "precision": None, "recall": None, "accuracy": None}
            f1s, ps, rs, accs = [], [], [], []
            for t in ts:
                y_hat = (y_score >= t).astype(int)
                tp = int(((y_hat == 1) & (y_true == pos_idx)).sum())
                fp = int(((y_hat == 1) & (y_true != pos_idx)).sum())
                fn = int(((y_hat == 0) & (y_true == pos_idx)).sum())
                tn = int(((y_hat == 0) & (y_true != pos_idx)).sum())
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                acc = (tp + tn) / len(y_true)
                f1s.append(f1); ps.append(precision); rs.append(recall); accs.append(acc)
                if f1 > best["f1"]:
                    best.update({"threshold": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall), "accuracy": float(acc)})
            fig = plt.figure(figsize=(5, 3))
            plt.plot(ts, f1s, label="F1")
            plt.plot(ts, ps, label="Precision")
            plt.plot(ts, rs, label="Recall")
            plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend(); plt.tight_layout()
            safe_pyplot(fig)
            st.json(best)

            import csv
            from io import StringIO
            sio = StringIO()
            w = csv.writer(sio)
            w.writerow(["threshold", "precision", "recall", "f1", "accuracy"])
            for i in range(len(ts)):
                w.writerow([f"{ts[i]:.3f}", f"{ps[i]:.6f}", f"{rs[i]:.6f}", f"{f1s[i]:.6f}", f"{accs[i]:.6f}"])
            st.download_button("Download threshold metrics (CSV)", data=sio.getvalue(), file_name="threshold_metrics.csv", mime="text/csv")

# =====================
# Mode: Export Grad-CAMs
# =====================
else:
    st.subheader("Export Grad-CAM overlays to ZIP")
    if datasets is None:
        st.error("Missing torchvision/data utils.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            split = st.selectbox("Split", ["train", "val", "test"], 2)
        with c2:
            subset = st.selectbox("Subset", ["All images", "Misclassified only"], 0)
        with c3:
            limit = st.slider("Max images", 10, 500, 100, 10)
        c4, c5 = st.columns(2)
        with c4:
            batch_size = st.selectbox("Batch size", [16, 32, 64], 1)
        with c5:
            gen = st.button("Generate & Download ZIP")

        if gen:
            split_dir = os.path.join(data_dir, split)
            out = scan_split(split_dir, batch_size=batch_size)
            if out is None:
                st.stop()
            paths, y_true, probs, y_pred = out
            if subset == "Misclassified only":
                idxs = [i for i in range(len(paths)) if int(y_true[i]) != int(y_pred[i])]
            else:
                idxs = list(range(len(paths)))
            idxs = idxs[:limit]

            target_layer = model.layer4[-1]
            cam = GradCAM(model, target_layer)

            memzip = io.BytesIO()
            with zipfile.ZipFile(memzip, "w", zipfile.ZIP_DEFLATED) as zf:
                for i in idxs:
                    img = Image.open(paths[i]).convert("RGB")
                    x = preprocess(img).to(device=device, dtype=TORCH_DTYPE)
                    target_idx = torch.tensor([int(y_pred[i])], dtype=torch.long, device=device)
                    x.requires_grad_(True)
                    with torch.enable_grad():
                        cam_map = cam(x, class_idx=target_idx).squeeze(0).squeeze(0).cpu().numpy()
                    fig = overlay_cam_on_image(x[0].detach().cpu(), cam_map)

                    buf = io.BytesIO()
                    fname = f"{split}_{i:05d}_true-{class_names[int(y_true[i])]}_pred-{class_names[int(y_pred[i])]}.png"
                    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    zf.writestr(fname, buf.read())

            cam.close()
            memzip.seek(0)
            st.success(f"Exported {len(idxs)} overlays.")
            st.download_button("Download Grad-CAMs (ZIP)", data=memzip.getvalue(), file_name=f"gradcams_{split}_{subset.replace(' ','_').lower()}.zip", mime="application/zip")
