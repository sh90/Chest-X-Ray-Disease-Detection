
IMG_SIZE = 224
TRAIN_IMG_SIZE = 256
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
DEFAULTS = {
    "epochs": 2,
    "freeze_epochs": 1,
    "batch_size": 16,
    "lr": 1e-3,
    "num_workers": 2,
    "out_dir": "outputs",
    "seed": 42,
}
