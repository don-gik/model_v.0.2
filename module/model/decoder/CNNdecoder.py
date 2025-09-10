import torch
import torch.nn as nn
import torch.nn.functional as F

from module.model.decoder.hf_refine_decoder import FreqRefineDecoder

class AutoDecoder(nn.Module):
    """
    Freeze된 ASTRA encoder 위에 붙는 간단한 복원 decoder.
    입력: [B, 1, C, H, W] → encoder → [B, D, H/2, W/2]
    출력: 복원된 [B, C, H, W] (target과 동일 shape)
    """
    def __init__(self, encoder, decoder, in_channels=256, out_channels=4):
        super().__init__()
        self.encoder = encoder

        self.decoder = decoder

    def forward(self, x):  # x: [B, 1, C, H, W]
        with torch.no_grad():
            z = self.encoder(x)  # [B, D, H/2, W/2]
        return self.decoder(z)  # [B, C, H, W]
