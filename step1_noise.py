import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from shared import get_device


# 載入圖片並轉成 tensor
def load_image(path, image_size=64):
    device = get_device()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # 將像素轉成 [0,1] 範圍的 tensor
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).to(device)

# 將圖片加上高斯噪聲
def add_noise(image_tensor, noise_level=1.0):
    noise = torch.randn_like(image_tensor) * noise_level
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)  # 保持在 [0,1] 範圍
    return noisy_image

# 顯示圖片
def show_image(tensor, title="Image"):
    image = tensor.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 範例用法
if __name__ == "__main__":
    image_path = "cop.jpg"  # 換成你的圖片路徑
    image = load_image(image_path)
    noisy_image = add_noise(image, noise_level=1)

    show_image(image, "Original")
    show_image(noisy_image, "Noisy")
