import torch

from shared import load_image, show_image


# 將圖片加上高斯噪聲
def add_noise(image_tensor, noise_level=1.0):
    noise = torch.randn_like(image_tensor) * noise_level
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)  # 保持在 [0,1] 範圍
    return noisy_image

# 範例用法
if __name__ == "__main__":
    image_path = "cop.jpg"  # 換成你的圖片路徑
    image = load_image(image_path)
    noisy_image = add_noise(image, noise_level=1)

    show_image(image, "Original")
    show_image(noisy_image, "Noisy")
