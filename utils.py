
import random, os, numpy as np, torch

"""
It’s a reproducibility helper. 
It seeds every major random-number generator your code might touch so runs are (as much as possible) repeatable.

random.seed(seed) → seeds Python’s built-in RNG (used by some torchvision transforms and any random.* calls).

np.random.seed(seed) → seeds NumPy’s RNG (if you ever use NumPy for shuffles/augments).

torch.manual_seed(seed) → seeds PyTorch’s CPU RNG (affects weight init, dropout, torch.rand*, etc.).

torch.cuda.manual_seed_all(seed) → seeds PyTorch’s GPU RNG on all visible CUDA devices (dropout, etc. on GPU).

os.environ["PYTHONHASHSEED"] = str(seed) → aims to fix Python’s hash randomization (affects dict/set ordering).
⚠Note: this env var must be set before the Python process starts to truly control hashing; setting it at runtime is mostly a no-op but harmless."""

def seed_everything(seed: int = 42):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
