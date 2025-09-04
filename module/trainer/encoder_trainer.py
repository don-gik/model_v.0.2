import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from module.model.encoder import Encoder
from module.dataset.ensodataset import ENSOSkipGramDataset
from torch.nn.utils import clip_grad_norm_
import wandb


def info_nce_loss(anchor, positives, negatives, temperature=0.1):
    anchor = anchor.unsqueeze(1)  # [B, 1, D]
    pos_sim = F.cosine_similarity(anchor, positives, dim=-1) / temperature  # [B, P]
    neg_sim = F.cosine_similarity(anchor, negatives, dim=-1) / temperature  # [B, N]
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, P+N]
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
    return F.cross_entropy(logits, labels)


def train_encoder(encoder, base_npz_data, epochs=100, batch_size=64, lr=1e-4, device='cuda'):
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    try:
        encoder.load_state_dict(torch.load("./models/astra_encoder.pth", map_location=device))
        print("✅ 체크포인트 로드 완료")
    except:
        print("🔁 새로 학습 시작")

    wandb.init(project="astra-encoder-infonce")

    for epoch in range(epochs):
        dataset = ENSOSkipGramDataset(base_npz_data, num_samples=2048)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        encoder.train()
        total_loss = 0

        for anchor, positive, negatives in loader:
            B, C, H, W = anchor.shape
            N = negatives.shape[1]

            anchor = anchor.unsqueeze(1).to(device)     # [B, 1, C, H, W]
            positive = positive.unsqueeze(1).to(device) # [B, 1, C, H, W]
            negatives = negatives.to(device)            # [B, N, C, H, W]
            negatives = negatives.reshape(B * N, C, H, W).unsqueeze(1)

            z_anchor = encoder(anchor).flatten(1)                   # [B, D]
            z_pos = encoder(positive).flatten(1).unsqueeze(1)       # [B, 1, D]
            z_neg = encoder(negatives).flatten(1).reshape(B, N, -1) # [B, N, D]

            loss = info_nce_loss(z_anchor, z_pos, z_neg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(loader)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.6f}")
        wandb.log({"Loss": total_loss, "epoch": epoch})
        torch.save(encoder.state_dict(), './models/astra_encoder.pth')


def main():
    import numpy as np
    npz = np.load('./data/enso_normalized.npz')
    base_npz_data = npz['data'][:, :, :40, :200]  # [T, C, H, W]
    encoder = Encoder(in_channels=6, embed_dim=256, mid_dim=128)
    train_encoder(encoder, base_npz_data, batch_size=32)
