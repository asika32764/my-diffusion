import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from shared import get_device

device = get_device()

# 定義 beta schedule（線性）
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T).to(device)

# 預先計算 alpha、alpha_bar
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

# 顯示圖片
def show_tensor_image(img_tensor, title=""):
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 執行範例
if __name__ == "__main__":
    T = 1000
    image_path = "cop.jpg"
    x_0 = load_image(image_path)  # shape: [1, 3, 64, 64]

    betas = get_beta_schedule(T)
    alphas, alpha_bars = get_alphas(betas)

    # 試一個中間 timestep，例如 t=200
    t = torch.tensor([200]).to(device)  # batch 一張圖也要有 batch 維度
    x_t, noise = q_sample(x_0, t, alpha_bars)

    # show_tensor_image(x_0, "Original x₀")
    show_tensor_image(x_t, f"Noised xₜ (t={t.item()})")
