import torch
import torch.nn as nn

from utils.unet import SinusoidalTimeEmbedding


# UNet Block
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv(x)
        t = self.time_mlp(t_emb)
        return h + t[:, :, None, None]


# 簡化版 UNet 模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.down1 = UNetBlock(in_channels, 64, time_emb_dim)
        self.down2 = UNetBlock(64, 128, time_emb_dim)

        self.middle = UNetBlock(128, 128, time_emb_dim)

        self.up1 = UNetBlock(128 + 128, 64, time_emb_dim)  # concat d2 + up(m)
        self.up2 = UNetBlock(64 + 64, in_channels, time_emb_dim)  # concat d1 + up(u1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        # downx 先增加通道，pool 再縮減長寬
        d1 = self.down1(x, t_emb)  # [B, 64, 64, 64]
        d2 = self.down2(self.pool(d1), t_emb)  # [B, 128, 32, 32]

        m = self.middle(self.pool(d2), t_emb)  # [B, 128, 16, 16]

        u1_input = torch.cat([self.upsample(m), d2], dim=1)  # concat on channel dim: 128+128
        u1 = self.up1(u1_input, t_emb)  # input: 256 → output: 64

        u2_input = torch.cat([self.upsample(u1), d1], dim=1)  # 64+64 = 128
        u2 = self.up2(u2_input, t_emb)  # input: 128 → output: 3

        return u2

if __name__ == "__main__":
    model = SimpleUNet()
    x = torch.randn(1, 3, 64, 64)  # 一張加了噪聲的圖片
    t = torch.tensor([500])        # 對應 timestep

    pred = model(x, t)
    print("預測 epsilon shape:", pred.shape)  # 應該是 [1, 3, 64, 64]
