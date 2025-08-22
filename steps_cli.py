
# main.py
# Orchestrates the full pipeline using modular pieces.
import os, argparse
import numpy as np
import torch

from config import DEFAULTS
from utils import seed_everything, device_auto, count_parameters
from data import build_datasets, build_dataloaders, show_batch_images
from modeling import build_model
from training import train_one_epoch, evaluate, plot_training
from evaluation import plot_confusion_and_curves
from gradcam import GradCAM, overlay_cam_on_image

import certifi
print(certifi.where())
import os

# Option 1: hard-code your cert bundle path
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]

# (optional) print to confirm
print("Using cert bundle:", os.environ["SSL_CERT_FILE"])


def toggle_backbone_trainable(model, flag: bool):
    for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters():
            p.requires_grad = flag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to chest_xray folder',default="chest_xray")
    parser.add_argument('--epochs', type=int, default=DEFAULTS["epochs"])
    parser.add_argument('--freeze_epochs', type=int, default=DEFAULTS["freeze_epochs"])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS["batch_size"])
    parser.add_argument('--lr', type=float, default=DEFAULTS["lr"])
    parser.add_argument('--num_workers', type=int, default=DEFAULTS["num_workers"])
    parser.add_argument('--out_dir', type=str, default=DEFAULTS["out_dir"])
    parser.add_argument('--seed', type=int, default=DEFAULTS["seed"])
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = device_auto()
    print("Device:", device)

    # DATA
    data = build_datasets(args.data_dir)
    loaders = build_dataloaders(data, batch_size=args.batch_size, num_workers=args.num_workers)
    class_names = list(data['train'].class_to_idx.keys())
    print("Classes:", class_names)

    # EDA: sample batch
    show_batch_images(loaders['train'], data['train'].class_to_idx, max_images=8,
                      savepath=os.path.join(args.out_dir, "sample_batch.png"))

    # MODEL
    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    print("Trainable params:", count_parameters(model))

    # Phase 1: freeze backbone
    """
    What’s happening

        Backbone vs head: The backbone is the pretrained feature extractor (e.g., all ResNet conv blocks). 
        The head is the new final classifier layer(s) you added for your task.
        toggle_backbone_trainable(model, False) sets requires_grad=False on the backbone params, so only the head updates.
        The optimizer is built on only trainable params (if p.requires_grad), i.e., just the head.
        You run a few epochs to let the head learn to read the already-useful features from ImageNet without disturbing them.
        Why do this
        Faster, stabler start on small/medical datasets.
        Avoids wrecking good pretrained features (“catastrophic forgetting”) before the head aligns to your labels.
        
        So basically,
        “We first train a tiny classifier head on top of frozen features. Think of it like keeping the ‘X-ray vocabulary’ fixed and just learning a new ‘sentence’ for pneumonia vs normal.”
"""
    toggle_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.freeze_epochs+1):
        print(f"\nEpoch {epoch}/{args.freeze_epochs} (frozen backbone)")
        tr_loss, tr_acc = train_one_epoch(model, loaders['train'], optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders['val'], device)
        print(f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
        history['train_loss'].append(tr_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc);   history['val_acc'].append(val_acc)

    # Phase 2: fine-tune all
    """Fine - tune the whole network """
    """
    What’s happening here?
    Now you unfreeze the backbone (requires_grad=True) so every layer can update.
    
    New optimizer over all parameters.
    Lower LR (* 0.5): fine-tuning uses a gentler step to avoid destroying useful pretrained features.
    Train remaining epochs; keep logging into history for plotting curves later.
    
    Why we are doing this?
    Once the head is reasonable, fine-tuning lets earlier layers adapt subtly to chest X-ray textures (very different from natural images), improving accuracy.
    So basically,
    “After the head learns the task, we let the earlier layers move a bit—small, careful updates—to specialize from ‘general images’ to ‘medical X-rays’.”
    """
    toggle_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    remaining = max(0, args.epochs - args.freeze_epochs)
    for i in range(remaining):
        epoch = args.freeze_epochs + i + 1
        print(f"\nEpoch {epoch}/{args.epochs} (fine-tuning)")
        tr_loss, tr_acc = train_one_epoch(model, loaders['train'], optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders['val'], device)
        print(f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
        history['train_loss'].append(tr_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc);   history['val_acc'].append(val_acc)

    # Curves
    plot_training(history, save_prefix=os.path.join(args.out_dir, "training_curves"))

    # Final Test
    test_loss, test_acc, y_true, y_score = evaluate(model, loaders['test'], device)
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(y_true, y_score)
    except Exception:
        test_auc = float('nan')
    print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.4f}, roc_auc={test_auc:.4f}")
    plot_confusion_and_curves(y_true, y_score, class_names, save_prefix=os.path.join(args.out_dir, "test"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "cxr_resnet18.pt"))
    print("Saved model to", os.path.join(args.out_dir, "cxr_resnet18.pt"))

    # Grad-CAM on a few test images
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    model.eval()
    images_done = 0
    with torch.no_grad():
        for x, _ in loaders['test']:
            x = x.to(device)
            logits = model(x)
            cls = logits.argmax(1)
            x.requires_grad_()
            cam_map = cam(x, class_idx=cls)  # [B,1,H,W]
            cam_map = cam_map.squeeze(1).cpu().numpy()
            for i in range(min(x.size(0), 6)):
                fig = overlay_cam_on_image(x[i].detach().cpu(), cam_map[i])
                out_path = os.path.join(args.out_dir, f"gradcam_{images_done+i:02d}.png")
                import matplotlib.pyplot as plt
                plt.savefig(out_path, dpi=140, bbox_inches='tight'); plt.close(fig)
            images_done += min(x.size(0), 6)
            break
    cam.close()
    print(f"Saved Grad-CAM overlays to {args.out_dir}/gradcam_*.png")

if __name__ == "__main__":
    main()
