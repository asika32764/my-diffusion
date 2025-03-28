import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from torchvision import transforms
from PIL import Image
import numpy as np

from shared import get_device

device = get_device()

# 建立 beta 與 alpha
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T).to(device)

def get_alphas(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars

def q_sample(x_0, t, alpha_bars):
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t

def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).to(device).unsqueeze(0)

# 把 tensor 轉 numpy image
def tensor_to_image(img_tensor):
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img, 0, 1)

# 產生動畫
def create_diffusion_animation(x_0, alpha_bars, filename="diffusion.gif", step=50):
    fig = plt.figure()
    ims = []

    for t in range(0, len(alpha_bars), step):
        t_tensor = torch.tensor([t])
        x_t = q_sample(x_0, t_tensor, alpha_bars)
        img = tensor_to_image(x_t)
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
    ani.save(filename, writer='pillow')
    plt.close()
    print(f"動畫已儲存為 {filename}")

# 執行
if __name__ == "__main__":
    T = 1000
    image_path = "cop.jpg"  # 換成你自己的圖片路徑
    x_0 = load_image(image_path)

    betas = get_beta_schedule(T)
    alphas, alpha_bars = get_alphas(betas)

    create_diffusion_animation(x_0, alpha_bars, filename="dist/diffusion.gif", step=50)
