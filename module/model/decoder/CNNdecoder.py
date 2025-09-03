import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoDecoder(nn.Module):
    """
    Freeze된 ASTRA encoder 위에 붙는 간단한 복원 decoder.
    입력: [B, 1, C, H, W] → encoder → [B, D, H/2, W/2]
    출력: 복원된 [B, C, H, W] (target과 동일 shape)
    """
    def __init__(self, encoder, in_channels=256, out_channels=4):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # [B, 256, 20, 100] → [B, 256, 40, 200]
            
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),  # 깊은 block 시작
            nn.GELU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # 최종 출력 (activation 없음)
        )

    def forward(self, x):  # x: [B, 1, C, H, W]
        with torch.no_grad():
            z = self.encoder(x)  # [B, D, H/2, W/2]
        return self.decoder(z)  # [B, C, H, W]
