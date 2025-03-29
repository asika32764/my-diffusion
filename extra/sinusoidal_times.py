import torch
import math
import matplotlib.pyplot as plt

def sinusoidal_time_embedding(t, dim):
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim) * -emb_scale)  # 頻率向量
    emb = torch.outer(t, freqs)  # t: [B]、freqs: [half_dim] → [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# 繪圖範例
dim = 64
t = torch.arange(0, 1000).float()  # 時間從 0 到 999
emb = sinusoidal_time_embedding(t, dim)  # [1000, dim]

plt.figure(figsize=(12, 6))
for i in range(8):  # 畫前 8 維
    plt.plot(t, emb[:, i], label=f"dim {i}")
plt.legend()
plt.title("Sinusoidal Time Embedding")
plt.xlabel("t (time step)")
plt.ylabel("embedding value")
plt.grid(True)
plt.show()
