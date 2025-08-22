
import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from config import NORM_MEAN, NORM_STD
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None; self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, m, inp, out): self.activations = out.detach()
    def _backward_hook(self, m, gin, gout): self.gradients = gout[0].detach()
    def __call__(self, x, class_idx=None):
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None: class_idx = logits.argmax(dim=1)
            loss = logits[torch.arange(logits.size(0)), class_idx].sum()
            self.model.zero_grad(); loss.backward()
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = (cam - cam.amin(dim=(2,3), keepdim=True)) / (cam.amax(dim=(2,3), keepdim=True) - cam.amin(dim=(2,3), keepdim=True) + 1e-6)
        return cam
    def close(self): self.hook_a.remove(); self.hook_g.remove()
def overlay_cam_on_image(img_tensor, cam_2d):
    img = img_tensor.permute(1,2,0).cpu().numpy()
    std = np.array(NORM_STD); mean = np.array(NORM_MEAN)
    img = np.clip(std*img + mean, 0, 1)
    plt.figure(figsize=(4,4)); plt.imshow(img); plt.imshow(cam_2d, cmap='jet', alpha=0.4); plt.axis('off')
    return plt.gcf()
