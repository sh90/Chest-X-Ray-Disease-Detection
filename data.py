
from typing import Dict
import os, math, numpy as np, matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from config import IMG_SIZE, TRAIN_IMG_SIZE, NORM_MEAN, NORM_STD

"""
Grayscale(…=3): Chest X-rays are 1-channel; we duplicate to 3 channels so ImageNet-pretrained models (expecting RGB) can be used.

Resize(256): Make images a bit larger first…

RandomResizedCrop(224, scale=(0.85,1.0)): …then randomly crop/zoom down to the final size. 
Adds translation/zoom augmentation while keeping most of the lungs (at least 85% of the area).

RandomHorizontalFlip(0.5): Left–right flip half the time. 
For a binary NORMAL vs PNEUMONIA task this is usually fine; don’t use if side (left/right) matters.

RandomRotation(7): Small ±7° tilt to mimic acquisition variation. Keeps realism; bigger angles can distort anatomy.

ToTensor(): Converts to PyTorch tensor, channels-first, scales to [0,1].

Normalize(mean,std): Per-channel (x-mean)/std with ImageNet values so inputs match what the pretrained backbone expects.

“Training uses mild randomness to help generalization (crop/flip/rotate).
Validation/Test are deterministic so metrics are comparable.”

“We convert X-rays to 3 channels and ImageNet-normalize because we’re using a pretrained ResNet.”

“TRAIN_IMG_SIZE (e.g., 256) + random crop to IMG_SIZE (e.g., 224) gives gentle augmentation without cutting off lungs.”

"""
def build_transforms(train: bool = True):
    if train:
        """
        This is your image preprocessing/augmentation pipeline. It runs one transform after another (that’s what Compose does)
        """
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((TRAIN_IMG_SIZE, TRAIN_IMG_SIZE)), #  e.g., 256×256
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)), # # random 85–100% area -> 224×224
            transforms.RandomHorizontalFlip(0.5), # 50% chance
            transforms.RandomRotation(7), # ±7° rotate
            transforms.ToTensor(), # HWC [0,255] -> CHW [0,1]
            transforms.Normalize(NORM_MEAN, NORM_STD), # ImageNet mean/std
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

def build_datasets(data_dir: str) -> Dict[str, datasets.ImageFolder]:
    data = {}
    for split in ["train", "val", "test"]:
        tfms = build_transforms(train=(split=="train"))
        folder = os.path.join(data_dir, split)
        data[split] = datasets.ImageFolder(folder, transform=tfms)
    return data

def _targets(ds):
    if hasattr(ds, "targets"): return ds.targets
    if hasattr(ds, "samples"): return [y for _,y in ds.samples]
    return [y for _,y in ds.imgs]

def build_dataloaders(data, batch_size=16, num_workers=2):
    tt = _targets(data["train"])
    import numpy as np
    class_counts = np.bincount(tt)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[tt]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights), replacement=True)
    return {
        "train": DataLoader(data["train"], batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(data["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(data["test"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }

def show_batch_images(dl: DataLoader, class_to_idx: dict, max_images=8, savepath=None):
    inv = {v:k for k,v in class_to_idx.items()}
    imgs, labels = next(iter(dl))
    n = min(max_images, imgs.size(0))
    cols = min(4, n); rows = int(np.ceil(n/cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        npimg = imgs[i].permute(1,2,0).cpu().numpy()
        std = np.array([0.229, 0.224, 0.225]); mean = np.array([0.485, 0.456, 0.406])
        npimg = std*npimg + mean; npimg = np.clip(npimg, 0, 1)
        plt.imshow(npimg); plt.title(inv[int(labels[i])]); plt.axis('off')
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()
