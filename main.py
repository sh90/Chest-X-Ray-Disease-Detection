
import os, argparse, torch
from config import DEFAULTS
from utils import seed_everything, device_auto, count_parameters
from data import build_datasets, build_dataloaders, show_batch_images
from modeling import build_model
from training import train_one_epoch, evaluate, plot_training
from evaluation import plot_confusion_and_curves
from gradcam import GradCAM, overlay_cam_on_image
def toggle_backbone(model, flag: bool):
    for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters(): p.requires_grad = flag
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=DEFAULTS["epochs"])
    p.add_argument('--freeze_epochs', type=int, default=DEFAULTS["freeze_epochs"])
    p.add_argument('--batch_size', type=int, default=DEFAULTS["batch_size"])
    p.add_argument('--lr', type=float, default=DEFAULTS["lr"])
    p.add_argument('--num_workers', type=int, default=DEFAULTS["num_workers"])
    p.add_argument('--out_dir', type=str, default=DEFAULTS["out_dir"])
    p.add_argument('--seed', type=int, default=DEFAULTS["seed"])
    p.add_argument('--save_model', action='store_true')
    args = p.parse_args()
    seed_everything(args.seed); os.makedirs(args.out_dir, exist_ok=True)
    device = device_auto(); print("Device:", device)
    data = build_datasets(args.data_dir)
    loaders = build_dataloaders(data, batch_size=args.batch_size, num_workers=args.num_workers)
    class_names = list(data['train'].class_to_idx.keys()); print("Classes:", class_names)
    show_batch_images(loaders['train'], data['train'].class_to_idx, 8, savepath=os.path.join(args.out_dir, "sample_batch.png"))
    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    print("Trainable params:", count_parameters(model))
    history={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    toggle_backbone(model, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for e in range(1, args.freeze_epochs+1):
        tr, ta = train_one_epoch(model, loaders['train'], opt, device)
        vl, va, _, _ = evaluate(model, loaders['val'], device)
        print(f"[frozen {e}/{args.freeze_epochs}] train_loss={tr:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")
        history['train_loss'].append(tr); history['val_loss'].append(vl); history['train_acc'].append(ta); history['val_acc'].append(va)
    toggle_backbone(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr*0.5)
    remaining = max(0, args.epochs-args.freeze_epochs)
    for i in range(remaining):
        e = args.freeze_epochs+i+1
        tr, ta = train_one_epoch(model, loaders['train'], opt, device)
        vl, va, _, _ = evaluate(model, loaders['val'], device)
        print(f"[finetune {e}/{args.epochs}] train_loss={tr:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")
        history['train_loss'].append(tr); history['val_loss'].append(vl); history['train_acc'].append(ta); history['val_acc'].append(va)
    plot_training(history, save_prefix=os.path.join(args.out_dir,"training_curves"))
    tl, ta, y_true, y_score = evaluate(model, loaders['test'], device)
    from sklearn.metrics import roc_auc_score
    try: auc = float(roc_auc_score(y_true, y_score))
    except Exception: auc = float('nan')
    print(f"Test: loss={tl:.4f} acc={ta:.4f} auc={auc:.4f}")
    plot_confusion_and_curves(y_true, y_score, class_names, save_prefix=os.path.join(args.out_dir,"test"))
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "cxr_resnet18.pt"))
        print("Saved model to", os.path.join(args.out_dir, "cxr_resnet18.pt"))
    target_layer = model.layer4[-1]; cam = GradCAM(model, target_layer)
    for x,_ in loaders['test']:
        x = x.to(device)
        with torch.no_grad(): cls = model(x).argmax(1)
        x.requires_grad_(True)
        with torch.enable_grad(): cam_map = cam(x, class_idx=cls).squeeze(1).cpu().numpy()
        import matplotlib.pyplot as plt
        for i in range(min(x.size(0), 6)):
            fig = overlay_cam_on_image(x[i].detach().cpu(), cam_map[i])
            out = os.path.join(args.out_dir, f"gradcam_{i:02d}.png")
            plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
        break
    cam.close(); print(f"Saved Grad-CAM overlays to {args.out_dir}")
if __name__ == "__main__":
    main()
