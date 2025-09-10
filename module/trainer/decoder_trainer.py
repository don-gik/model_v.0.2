import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb

from module.model.encoder import Encoder
from module.dataset.ensodataset import ENSOSkipGramDataset
from module.model.decoder import AutoDecoder, FreqRefineDecoder


def spectral_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # pred/true: (B, T, C, H, W)
    X = torch.fft.rfftn(pred, dim=(-2, -1))
    Y = torch.fft.rfftn(true, dim=(-2, -1))
    return torch.mean(torch.abs(X - Y))


def _panel_pred_vs_target(pred: torch.Tensor, target: torch.Tensor, item: int = 0, max_channels: int = 3):
    with torch.no_grad():
        pred_np   = pred[item].detach().cpu().numpy()   # [C, H, W]
        target_np = target[item].detach().cpu().numpy() # [C, H, W]
        C, H, W = pred_np.shape
        show_c = min(C, max_channels)

        fig, axes = plt.subplots(show_c, 3, figsize=(9, 3 * show_c))
        if show_c == 1:
            axes = np.expand_dims(axes, axis=0)

        for c in range(show_c):
            p = pred_np[c]; t = target_np[c]; diff = p - t
            vmin = np.percentile(np.concatenate([p.ravel(), t.ravel()]), 1)
            vmax = np.percentile(np.concatenate([p.ravel(), t.ravel()]), 99)

            axes[c, 0].imshow(t, vmin=vmin, vmax=vmax); axes[c, 0].set_title(f"Ch{c} Target"); axes[c, 0].axis("off")
            axes[c, 1].imshow(p, vmin=vmin, vmax=vmax); axes[c, 1].set_title(f"Ch{c} Pred");   axes[c, 1].axis("off")
            d_abs = np.percentile(np.abs(diff.ravel()), 99)
            axes[c, 2].imshow(diff, vmin=-d_abs, vmax=+d_abs); axes[c, 2].set_title(f"Ch{c} Diff"); axes[c, 2].axis("off")

        plt.tight_layout()
        return fig


def _per_channel_rmse(pred: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        rmse = torch.sqrt(((pred - target) ** 2).mean(dim=(0, 2, 3)))
        return rmse.detach().cpu().numpy()


def main():
    wandb.init(project='Decoder Training')
    wandb.config.update({
        "batch_size": 32,
        "epochs": 250,
        "lr": 1e-4,
        "encoder_ckpt": "./models/astra_encoder.pth"
    })

    force_cpu = os.environ.get("USE_CPU_ONLY", "0") == "1"
    device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 250

    # --- data ---
    npz = np.load("./data/enso_normalized.npz")
    base_data = npz['data'][:, :, :40, :200]
    dataset = ENSOSkipGramDataset(base_data, num_samples=2048*4)
    num_workers = 4 if device.type == 'cuda' else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    # --- model ---
    encoder = Encoder(in_channels=4, embed_dim=128, mid_dim=32).to(device)
    encoder.load_state_dict(torch.load("./models/astra_encoder.pth", map_location=device))
    decoder_model = AutoDecoder(encoder, freeze_encoder=False).to(device)

    base_dec = decoder_model.decoder
    decoder_model.decoder = FreqRefineDecoder(
        base_decoder=base_dec,
        in_channels=encoder.out_dim,
        out_channels=4,
        cond_z=True,
        refine_width=96,
        refine_depth=3,
        dropout=0.05,
    ).to(device)

    # ---- critical: unify dtype/device for optimizer safety ----
    decoder_model.decoder.to(device=device, dtype=torch.float32)
    for m in decoder_model.decoder.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            m.float()

    # --- optimizer (no fused) ---
    optimizer = optim.AdamW(decoder_model.decoder.parameters(), lr=1e-4, fused=False, foreach=True)
    criterion = nn.MSELoss()

    wandb.watch(decoder_model.decoder, log="all", log_freq=100)

    # --- checkpoints (best-effort) ---
    loaded = False
    for ck in ["./models/auto_decoder_hf.pth", "./models/auto_decoder.pth"]:
        try:
            checkpoint = torch.load(ck, map_location=device)
            decoder_model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 체크포인트 로드됨: {ck}")
            loaded = True
            break
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ 로드 실패 ({ck}): {e}")

    if not loaded:
        try:
            base_sd = torch.load("./models/decoder.pth", map_location=device)
            if isinstance(decoder_model.decoder, FreqRefineDecoder):
                decoder_model.decoder.base.load_state_dict(base_sd, strict=False)
            else:
                decoder_model.decoder.load_state_dict(base_sd, strict=False)
            print("✅ 기본 디코더 가중치 로드됨 (decoder.pth)")
        except FileNotFoundError:
            print("🔁 체크포인트 없음. 새로 학습 시작")
        except Exception as e:
            print(f"⚠️ 기본 디코더 로드 실패: {e}")

    decoder_model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        total_loss = 0.0
        vis_anchor_cpu = None  # 첫 배치 일부를 저장해 epoch-end 시각화에 재사용

        for step, (anchor, _, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if vis_anchor_cpu is None:
                vis_anchor_cpu = anchor[:8].detach().cpu()

            x = anchor.unsqueeze(1).to(device, non_blocking=True)  # [B, 1, C, H, W]
            target = anchor.to(device, non_blocking=True)          # [B, C, H, W]

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                pred = decoder_model(x)  # [B, C, H, W]
                loss = criterion(pred, target)
                # spec_w=0.0 → 사용 안 함 (GPU 메모리 절약)
                # 필요 시: loss = loss + spec_w * spectral_loss(pred.unsqueeze(1), target.unsqueeze(1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                rmse = _per_channel_rmse(pred, target)
                wandb.log({
                    "train/step_loss": float(loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    **{f"rmse/ch{c}": float(v) for c, v in enumerate(rmse)}
                })

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": float(avg_loss), "epoch": epoch + 1})

        # --- epoch visualize using saved first batch ---
        with torch.no_grad():
            anchor_vis = vis_anchor_cpu.to(device)
            x_vis = anchor_vis.unsqueeze(1)
            t_vis = anchor_vis
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                p_vis = decoder_model(x_vis)

            fig = _panel_pred_vs_target(p_vis, t_vis, item=0, max_channels=3)
            wandb.log({"viz/pred_vs_target": wandb.Image(fig), "epoch": epoch + 1})
            plt.close(fig)

            rmse_epoch = _per_channel_rmse(p_vis, t_vis)
            wandb.log({**{f"rmse_epoch/ch{c}": float(v) for c, v in enumerate(rmse_epoch)},
                       "epoch": epoch + 1})

        # --- save checkpoints ---
        os.makedirs("./models", exist_ok=True)
        if isinstance(decoder_model.decoder, FreqRefineDecoder):
            torch.save(decoder_model.decoder.base.state_dict(), "./models/decoder.pth")
            torch.save(decoder_model.decoder.state_dict(), "./models/decoder_hf.pth")
            torch.save(decoder_model.state_dict(), "./models/auto_decoder_hf.pth")
        else:
            torch.save(decoder_model.decoder.state_dict(), "./models/decoder.pth")
            torch.save(decoder_model.state_dict(), "./models/auto_decoder.pth")


if __name__ == "__main__":
    main()
