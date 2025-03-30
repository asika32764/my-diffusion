import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from torchvision import transforms
from PIL import Image
import numpy as np

from shared import get_device, q_sample, load_image, get_beta_schedule, get_alphas, tensor_to_image

device = get_device()

# 產生動畫
def create_diffusion_animation(x_0, alpha_bars, filename="diffusion.gif", step=50):
    fig = plt.figure()
    ims = []

    for t in range(0, len(alpha_bars), step):
        t_tensor = torch.tensor([t])
        x_t, _ = q_sample(x_0, t_tensor, alpha_bars)
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
