
from typing import Dict
import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from tqdm import tqdm
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter=0.0; correct=0; total=0
    all_probs=[]; all_labels=[]
    for x,y in tqdm(loader, desc="Eval", leave=False):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_meter += loss.item()*x.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        correct += (preds==y).sum().item(); total += y.size(0)
        all_probs.append(probs[:,1].cpu().numpy()); all_labels.append(y.cpu().numpy())
    return loss_meter/total, correct/total, np.concatenate(all_labels), np.concatenate(all_probs)
def train_one_epoch(model, loader, optimizer, device):
    model.train(); loss_meter=0.0; correct=0; total=0
    for x,y in tqdm(loader, desc="Train", leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward(); optimizer.step()
        loss_meter += loss.item()*x.size(0)
        correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_meter/total, correct/total
def plot_training(history, save_prefix: str = None, show: bool | None = None):
    import os
    if show is None:
        show = os.environ.get("CXR_SHOW_PLOTS", "0") == "1"
    epochs = range(1, len(history['train_loss'])+1)
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss over Epochs'); plt.legend(); plt.tight_layout()
    if save_prefix: plt.savefig(save_prefix + '_loss.png', dpi=140, bbox_inches='tight')
    (plt.show() if show else plt.close())

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy over Epochs'); plt.legend(); plt.tight_layout()
    if save_prefix: plt.savefig(save_prefix + '_acc.png', dpi=140, bbox_inches='tight')
    (plt.show() if show else plt.close())
