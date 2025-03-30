import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from shared import get_device, load_image, get_beta_schedule, get_alphas, q_sample, show_image

device = get_device()

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
    show_image(x_t, f"Noised xₜ (t={t.item()})")
