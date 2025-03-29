import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (M1/M2)
    else:
        return torch.device("cpu")

device = get_device()

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T).to(device)

def get_alphas(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars

# 前向過程 q(x_t | x_0)
def q_sample(x_0, t, alpha_bars):
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

# 載入圖片
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return (transform(image)
            .to(device)
            .unsqueeze(0))  # 加 batch 維度

def tensor_to_image(img_tensor):
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img, 0, 1)

def show_image(tensor, title="Image"):
    # image = tensor.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    image = tensor_to_image(tensor)  # C,H,W -> H,W,C
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()