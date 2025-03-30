import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from s5_unet_simple import SimpleUNet
from shared import get_device

# === 預設參數 ===
T = 1000
IMAGE_PATH = "cop.jpg"
IMAGE_SIZE = 64
EPOCHS = 1000
DEVICE = get_device()

# === 1. 載入圖像 ===
def load_image(path, size=64):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

# === 2. beta & alpha schedule ===
def get_diffusion_schedule(T):
    betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

# === 3. get training sample ===
def get_training_sample(x_0, alpha_bars, T):
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=DEVICE)
    epsilon = torch.randn_like(x_0)
    alpha_bar_t = alpha_bars[t].reshape(B, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
    return x_t, t, epsilon

# === 4. 顯示圖片 ===
def show_tensor_image(img_tensor, title=""):
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# === 5. 匯入你的 UNet ===
# 請貼上你剛剛修正好的 SimpleUNet class 這裡！

# === 6. 開始訓練 ===
def train():
    # 準備資料與模型
    x_0 = load_image(IMAGE_PATH, IMAGE_SIZE)
    betas, alphas, alpha_bars = get_diffusion_schedule(T)
    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()

        # 產生 training sample
        x_t, t, epsilon = get_training_sample(x_0, alpha_bars, T)

        # 預測 epsilon
        pred_epsilon = model(x_t, t)

        # 計算損失
        loss = loss_fn(pred_epsilon, epsilon)

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            show_tensor_image(x_t, f"Noised Image (t={t.item()})")
            show_tensor_image(pred_epsilon, "Predicted ε")

    print("訓練完成 ✅")

if __name__ == "__main__":
    train()