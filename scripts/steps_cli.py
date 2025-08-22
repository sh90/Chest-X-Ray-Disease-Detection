
import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
from config import DEFAULTS
from utils import seed_everything, device_auto, count_parameters
from data import build_datasets, build_dataloaders, show_batch_images, build_transforms
from modeling import build_model
from training import train_one_epoch, evaluate, plot_training
from evaluation import plot_confusion_and_curves
from gradcam import GradCAM, overlay_cam_on_image

def cmd_prep(a):
    data = build_datasets(a.data_dir); print("Classes:", list(data['train'].class_to_idx.keys()))
    for split in ['train','val','test']:
        ds = data[split]; counts={k:0 for k in ds.class_to_idx.keys()}
        if hasattr(ds,"samples"): targets=[y for _,y in ds.samples]
        elif hasattr(ds,"targets"): targets=ds.targets
        else: targets=[y for _,y in ds.imgs]
        for t in targets:
            lab = list(ds.class_to_idx.keys())[list(ds.class_to_idx.values()).index(t)]
            counts[lab]+=1
        print(f"[{split}] total={len(ds)} :: "+", ".join([f"{k}={v}" for k,v in counts.items()]))
def cmd_dataloader(a):
    os.makedirs(a.out_dir, exist_ok=True)
    data=build_datasets(a.data_dir); loaders=build_dataloaders(data, batch_size=a.batch_size, num_workers=a.num_workers)
    show_batch_images(loaders['train'], data['train'].class_to_idx, 8, savepath=os.path.join(a.out_dir,"sample_batch.png"))
    print("Saved sample batch.")
def cmd_train(a):
    seed_everything(a.seed); os.makedirs(a.out_dir, exist_ok=True); dev=device_auto(); print("Device:", dev)
    data=build_datasets(a.data_dir); loaders=build_dataloaders(data, batch_size=a.batch_size, num_workers=a.num_workers)
    cls=list(data['train'].class_to_idx.keys()); model=build_model(num_classes=len(cls), pretrained=not a.no_pretrained).to(dev); print("Params:", count_parameters(model))
    def tog(flag):
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
            for p in m.parameters(): p.requires_grad=flag
    hist={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    tog(False); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    for e in range(a.freeze_epochs):
        tr,ta=train_one_epoch(model, loaders['train'], opt, dev); vl,va,_,_=evaluate(model, loaders['val'], dev)
        print(f"[frozen {e+1}/{a.freeze_epochs}] train={tr:.4f}/{ta:.4f} val={vl:.4f}/{va:.4f}")
        hist['train_loss'].append(tr); hist['val_loss'].append(vl); hist['train_acc'].append(ta); hist['val_acc'].append(va)
    tog(True); opt=torch.optim.AdamW(model.parameters(), lr=a.lr*0.5); rem=max(0,a.epochs-a.freeze_epochs)
    for i in range(rem):
        e=a.freeze_epochs+i+1; tr,ta=train_one_epoch(model, loaders['train'], opt, dev); vl,va,_,_=evaluate(model, loaders['val'], dev)
        print(f"[finetune {e}/{a.epochs}] train={tr:.4f}/{ta:.4f} val={vl:.4f}/{va:.4f}")
        hist['train_loss'].append(tr); hist['val_loss'].append(vl); hist['train_acc'].append(ta); hist['val_acc'].append(va)
    plot_training(hist, save_prefix=os.path.join(a.out_dir,"training_curves"))
    if a.save_model:
        torch.save(model.state_dict(), os.path.join(a.out_dir, "cxr_resnet18.pt")); print("Saved model.")
def cmd_eval(a):
    os.makedirs(a.out_dir, exist_ok=True); dev=device_auto()
    data=build_datasets(a.data_dir); loaders=build_dataloaders(data, batch_size=a.batch_size, num_workers=a.num_workers)
    cls=list(data['train'].class_to_idx.keys()); model=build_model(num_classes=len(cls), pretrained=False).to(dev)
    if not a.weights or not os.path.exists(a.weights): print("ERROR: provide --weights"); return
    state=torch.load(a.weights, map_location="cpu"); model.load_state_dict(state)
    tl,ta,y_true,y_score = evaluate(model, loaders['test'], dev); from sklearn.metrics import roc_auc_score
    try: auc=float(roc_auc_score(y_true,y_score))
    except Exception: auc=float("nan")
    print({"test_loss":tl,"test_acc":ta,"roc_auc":auc}); plot_confusion_and_curves(y_true,y_score,cls, save_prefix=os.path.join(a.out_dir,"test"))
def cmd_cam(a):
    os.makedirs(a.out_dir, exist_ok=True); dev=device_auto()
    data=build_datasets(a.data_dir); loaders=build_dataloaders(data, batch_size=a.batch_size, num_workers=a.num_workers)
    cls=list(data['train'].class_to_idx.keys()); model=build_model(num_classes=len(cls), pretrained=False).to(dev)
    if not a.weights or not os.path.exists(a.weights): print("ERROR: provide --weights"); return
    state=torch.load(a.weights, map_location="cpu"); model.load_state_dict(state); model.eval()
    target=model.layer4[-1]; cam=GradCAM(model,target); done=0
    for x,_ in loaders['test']:
        x=x.to(dev); 
        with torch.no_grad(): cls_idx=model(x).argmax(1)
        x.requires_grad_(True)
        with torch.enable_grad(): cam_map=cam(x, class_idx=cls_idx).squeeze(1).cpu().numpy()
        for i in range(min(x.size(0), a.num-done)):
            fig=overlay_cam_on_image(x[i].detach().cpu(), cam_map[i]); import matplotlib.pyplot as plt
            out=os.path.join(a.out_dir, f"gradcam_{done+i:02d}.png"); plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
        done += min(x.size(0), a.num-done)
        if done>=a.num: break
    cam.close(); print(f"Saved {done} overlays to {a.out_dir}")
def cmd_tune(a):
    import csv
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    os.makedirs(a.out_dir, exist_ok=True); dev=device_auto()
    tfm=build_transforms(train=False); from torchvision import datasets; from torch.utils.data import DataLoader
    ds=datasets.ImageFolder(os.path.join(a.data_dir,"test"), transform=tfm); ld=DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    cls=ds.classes; model=build_model(num_classes=len(cls), pretrained=False).to(dev); st=torch.load(a.weights, map_location="cpu"); model.load_state_dict(st); model.eval()
    y_true=[]; y_score=[]; import numpy as np
    with torch.no_grad():
        for x,y in ld:
            x=x.to(dev); y_true.extend(y.numpy().tolist()); y_score.extend(torch.softmax(model(x), dim=1)[:, a.pos_idx].cpu().numpy().tolist())
    y_true=np.array(y_true); y_score=np.array(y_score)
    ts=np.linspace(0,1,201); best={"threshold":None,"f1":-1,"precision":None,"recall":None,"accuracy":None}; rows=[["threshold","precision","recall","f1","accuracy"]]
    for t in ts:
        y_hat=(y_score>=t).astype(int); p=precision_score(y_true==a.pos_idx, y_hat, zero_division=0); r=recall_score(y_true==a.pos_idx, y_hat, zero_division=0); f1=f1_score(y_true==a.pos_idx, y_hat, zero_division=0); acc=accuracy_score(y_true==a.pos_idx, y_hat)
        rows.append([f"{t:.3f}",f"{p:.6f}",f"{r:.6f}",f"{f1:.6f}",f"{acc:.6f}"])
        if f1>best["f1"]: best.update({"threshold":float(t),"f1":float(f1),"precision":float(p),"recall":float(r),"accuracy":float(acc)})
    print("Best:", best); csv_path=os.path.join(a.out_dir,"threshold_metrics.csv"); 
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerows(rows)
    print("Saved CSV:", csv_path)
def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd", required=True)
    def add_common(sp):
        sp.add_argument("--data_dir", type=str, required=True)
        sp.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
        sp.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
        sp.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
        sp.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    sp=sub.add_parser("prep"); add_common(sp); sp.set_defaults(func=cmd_prep)
    sp=sub.add_parser("dataloader"); add_common(sp); sp.set_defaults(func=cmd_dataloader)
    sp=sub.add_parser("train"); add_common(sp); sp.add_argument("--epochs", type=int, default=DEFAULTS["epochs"]); sp.add_argument("--freeze_epochs", type=int, default=DEFAULTS["freeze_epochs"]); sp.add_argument("--lr", type=float, default=DEFAULTS["lr"]); sp.add_argument("--no_pretrained", action="store_true"); sp.add_argument("--save_model", action="store_true"); sp.set_defaults(func=cmd_train)
    sp=sub.add_parser("eval"); add_common(sp); sp.add_argument("--weights", type=str, required=True); sp.set_defaults(func=cmd_eval)
    sp=sub.add_parser("cam"); add_common(sp); sp.add_argument("--weights", type=str, required=True); sp.add_argument("--num", type=int, default=8); sp.set_defaults(func=cmd_cam)
    sp=sub.add_parser("tune"); add_common(sp); sp.add_argument("--weights", type=str, required=True); sp.add_argument("--pos_idx", type=int, default=1); sp.set_defaults(func=cmd_tune)
    return p
def main():
    a=parser().parse_args(); seed_everything(a.seed); a.func(a)
if __name__=="__main__": main()
