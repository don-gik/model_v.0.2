import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

from module.model.encoder import Encoder
from module.dataset.ensodataset import ENSOSkipGramDataset

import logging


try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s : %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)


# ----------------------------
# Loss
# ----------------------------
def info_nce_loss(z_anchor, z_pos, z_neg, temperature: float = 0.1):
    """
    z_anchor: [B, D]
    z_pos:    [B, P, D]
    z_neg:    [B, N, D]
    """
    # cosine similarity is already L2-normalized dot
    anchor = z_anchor.unsqueeze(1)                          # [B, 1, D]
    pos_sim = F.cosine_similarity(anchor, z_pos, dim=-1)    # [B, P]
    neg_sim = F.cosine_similarity(anchor, z_neg, dim=-1)    # [B, N]

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # [B, P+N]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# ----------------------------
# Train
# ----------------------------
def train_encoder(
    encoder: torch.nn.Module,
    base_npz_data: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    temperature: float = 0.1,
    num_workers: int = 4,
    device: str = "cuda",
    ckpt_dir: str = "./models",
):
    torch.backends.cudnn.benchmark = True
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.05)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Try resume
    last_ckpt = Path(ckpt_dir) / "astra_encoder_last.pth"
    best_ckpt = Path(ckpt_dir) / "astra_encoder_best.pth"
    best_loss = float("inf")
    if last_ckpt.exists():
        encoder.load_state_dict(torch.load(last_ckpt, map_location=device))
        logging.info("Resumed from last checkpoint")

    if _WANDB:
        wandb.init(project="astra-encoder-infonce", reinit=True, config=dict(
            lr=lr, batch_size=batch_size, epochs=epochs, temperature=temperature
        ))

    for epoch in range(1, epochs + 1):
        # Resample pairs each epoch
        dataset = ENSOSkipGramDataset(base_npz_data, num_samples=2048)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        encoder.train()
        running = 0.0

        for anchor, positive, negatives in loader:
            # anchor/positive: [B, C, H, W], negatives: [B, N, C, H, W]
            B, C, H, W = anchor.shape
            N = negatives.shape[1]

            # Encoder expects [B, 1, C, H, W]
            anchor = anchor.unsqueeze(1).to(device, non_blocking=True)
            positive = positive.unsqueeze(1).to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True).reshape(B * N, C, H, W).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else None):
                z_anchor = encoder(anchor).flatten(1)                 # [B, D]
                z_pos    = encoder(positive).flatten(1).unsqueeze(1)  # [B, 1, D]
                z_neg    = encoder(negatives).flatten(1).reshape(B, N, -1)  # [B, N, D]

                loss = info_nce_loss(z_anchor, z_pos, z_neg, temperature=temperature)

            scaler.scale(loss).backward()
            clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        epoch_loss = running / max(1, len(loader))
        logging.info(f"[Epoch {epoch:03d}] loss={epoch_loss:.6f}")

        if _WANDB:
            wandb.log({"loss": epoch_loss, "epoch": epoch})

        # Save last and best
        torch.save(encoder.state_dict(), last_ckpt)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), best_ckpt)

            logging.info(f"Best model is changed. Lowest Loss : {epoch_loss}")

    if _WANDB:
        wandb.finish()


# ----------------------------
# Main
# ----------------------------
def main():
    npz = np.load("./data/enso_normalized.npz")
    base_npz_data = npz["data"][:, :, :40, :200]  # [T, C, H, W]

    encoder = Encoder()
    train_encoder(
        encoder,
        base_npz_data,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        temperature=0.1,
        num_workers=4,
        device="cuda",
        ckpt_dir="./models/encoder",
    )


if __name__ == "__main__":
    main()
