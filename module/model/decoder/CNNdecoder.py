import torch
import torch.nn as nn
from module.model.decoder.decoder import Decoder  # 위 파일

# CNNdecoder.py
class AutoDecoder(nn.Module):
    def __init__(self, encoder, decoder=None, out_channels=4, freeze_encoder=True,
                 hidden=64, num_res=2, dropout=0.0, use_deconv=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        # encoder가 노출하는 채널 수 가져오기
        D = getattr(encoder, "out_dim", None) or getattr(encoder, "embed_dim", None)
        if D is None:
            # 최후수단: 더미 입력으로 추론 (H,W는 실제 값으로 대체)
            dev = next(self.encoder.parameters()).device
            dummy = torch.zeros(1, 1, out_channels, 40, 200, device=dev)  # H,W 맞게
            with torch.no_grad():
                D = self.encoder(dummy).shape[1]

        # ★ 여기서 D를 넘긴다!
        self.decoder = decoder if decoder is not None else Decoder(
            in_channels=D, out_channels=out_channels,
            hidden=hidden, num_res=num_res, dropout=dropout, use_deconv=use_deconv
        )

    def forward(self, x):
        # Conditionally stop gradients if encoder is frozen
        if all(not p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                z = self.encoder(x)        # [B, D, H/2, W/2]
        else:
            z = self.encoder(x)
        return self.decoder(z)
