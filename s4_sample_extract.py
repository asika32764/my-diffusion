import torch
import random

from shared import get_device, load_image, get_alphas
from step2_forward_process import get_beta_schedule

device = get_device()

# 訓練樣本產生器
def get_training_sample(x_0, alpha_bars, T):
    """
    x_0: 一張圖片 tensor，shape: [B, C, H, W]
    alpha_bars: 預先計算的 ᾱₜ，shape: [T]
    T: 總 timestep 數
    """

    # 隨機選一個 timestep t
    t = torch.randint(0, T, (x_0.shape[0],), device=x_0.device)  # 每張圖都獨立選 t

    # 產生隨機噪聲 ε
    epsilon = torch.randn_like(x_0)

    # 根據 t 計算 x_t
    alpha_bar_t = alpha_bars[t].reshape(-1, 1, 1, 1)  # 對應每張圖的 t == [: None, None, None]
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

    return x_t, t, epsilon

if __name__ == "__main__":
    # 準備資料
    x_0 = load_image("cop.jpg")  # 換你的圖片
    T = 1000
    betas = get_beta_schedule(T)
    _, alpha_bars = get_alphas(betas)

    # 測試 sample
    x_t, t, epsilon = get_training_sample(x_0, alpha_bars, T)
    print(f"x_t shape: {x_t.shape}, t: {t}, epsilon shape: {epsilon.shape}")
