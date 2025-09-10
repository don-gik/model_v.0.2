import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from module.model.encoder import Encoder
from module.dataset.ensodataset import ENSOSkipGramDataset
from module.model.decoder import AutoDecoder, FreqRefineDecoder
import wandb
import os


def spectral_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # pred/true: (B, T, C, H, W)
    # 공간축에 대해 rfftn
    X = torch.fft.rfftn(pred, dim=(-2, -1))
    Y = torch.fft.rfftn(true, dim=(-2, -1))
    return torch.mean(torch.abs(X - Y))

# --- NEW: add imports ---
import numpy as np
import matplotlib.pyplot as plt
import wandb

# --- NEW: nice panel for (target, pred, diff) ---
def _panel_pred_vs_target(pred: torch.Tensor, target: torch.Tensor, item: int = 0, max_channels: int = 3):
    """
    pred/target: [B, C, H, W]
    하나의 배치 샘플(item)에 대해 상위 max_channels개 채널을 (Target | Pred | Diff) 3열 패널로 시각화
    """
    with torch.no_grad():
        pred_np   = pred[item].detach().cpu().numpy()   # [C, H, W]
        target_np = target[item].detach().cpu().numpy() # [C, H, W]

        C, H, W = pred_np.shape
        show_c = min(C, max_channels)

        fig, axes = plt.subplots(show_c, 3, figsize=(9, 3 * show_c))
        if show_c == 1:
            axes = np.expand_dims(axes, axis=0)  # shape 통일

        for c in range(show_c):
            p = pred_np[c]
            t = target_np[c]
            diff = p - t

            # 컬러 스케일을 pred/target 공통으로 맞춤 (퍼센타일 기반으로 이상치 완화)
            vmin = np.percentile(np.concatenate([p.ravel(), t.ravel()]), 1)
            vmax = np.percentile(np.concatenate([p.ravel(), t.ravel()]), 99)

            axes[c, 0].imshow(t, vmin=vmin, vmax=vmax)
            axes[c, 0].set_title(f"Ch{c} Target"); axes[c, 0].axis("off")

            axes[c, 1].imshow(p, vmin=vmin, vmax=vmax)
            axes[c, 1].set_title(f"Ch{c} Pred"); axes[c, 1].axis("off")

            # 차이는 0 중심으로 보기 좋게
            d_abs = np.percentile(np.abs(diff.ravel()), 99)
            axes[c, 2].imshow(diff, vmin=-d_abs, vmax=+d_abs)
            axes[c, 2].set_title(f"Ch{c} Diff"); axes[c, 2].axis("off")

        plt.tight_layout()
        return fig

# --- (선택) 채널별 RMSE 계산 ---
def _per_channel_rmse(pred: torch.Tensor, target: torch.Tensor):
    # [B, C, H, W] -> [C]
    with torch.no_grad():
        rmse = torch.sqrt(((pred - target) ** 2).mean(dim=(0, 2, 3)))
        return rmse.detach().cpu().numpy()


def main():
    wandb.init(project='Decoder Training')
    # (선택) 러닝 설정 기록
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

    npz = np.load("./data/enso_normalized.npz")
    base_data = npz['data'][:, :, :40, :200]
    dataset = ENSOSkipGramDataset(base_data, num_samples=2048*4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
        prefetch_factor=4,
    )

    encoder = Encoder(in_channels=4, embed_dim=128, mid_dim=32).to(device)
    encoder.load_state_dict(torch.load("./models/astra_encoder.pth", map_location=device))
    # Build base auto-decoder (encoder frozen by default)
    decoder_model = AutoDecoder(encoder, freeze_encoder=False).to(device)
    # Wrap its decoder with high-frequency refinement (keeps old decoder intact)
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

    # Fused Adam when available
    try:
        optimizer = optim.AdamW(decoder_model.decoder.parameters(), lr=1e-4, fused=True)
    except TypeError:
        optimizer = optim.AdamW(decoder_model.decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # --- NEW: wandb.watch로 grad/weights 자동 로깅 ---
    wandb.watch(decoder_model.decoder, log="all", log_freq=100)

    # --- checkpoint 로드 ---
    # Load checkpoints with backward compatibility
    loaded = False
    for ck in ["./models/auto_decoder_hf.pth", "./models/auto_decoder.pth"]:
        try:
            checkpoint = torch.load(ck, map_location=device)
            decoder_model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 체크포인트 로드됨: {ck}")
            loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ 로드 실패 ({ck}): {e}")
    # Optionally load base decoder only if available
    if not loaded:
        try:
            base_sd = torch.load("./models/decoder.pth", map_location=device)
            # If wrapped, load into the base submodule
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
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for epoch in range(epochs):
        total_loss = 0.0

        for step, (anchor, _, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            x = anchor.unsqueeze(1).to(device, non_blocking=True)  # [B, 1, C, H, W]
            target = anchor.to(device, non_blocking=True)          # [B, C, H, W]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = decoder_model(x)             # [B, C, H, W]
                # Avoid computing spectral loss unless weighted > 0 to save GPU mem
                spec_w = 0.0
                loss = criterion(pred, target)
                if spec_w > 0.0:
                    loss = loss + spec_w * spectral_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # --- NEW: step별 (드문 간격) 미니 로깅 ---
            if step % 50 == 0:
                # 채널별 RMSE (현재 배치 기준)
                rmse = _per_channel_rmse(pred, target)
                ch_logs = {f"rmse/ch{c}": float(v) for c, v in enumerate(rmse)}
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    **ch_logs
                })

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # --- epoch 스칼라 로깅 ---
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        # --- NEW: epoch마다 시각화 이미지 로깅 (첫 배치만 재사용; 비용 절감하려면 간격 조절) ---
        # dataloader에서 샘플 하나 뽑아 재시각화
        with torch.no_grad():
            anchor_vis, _, _ = next(iter(dataloader))
            x_vis = anchor_vis.unsqueeze(1).to(device, non_blocking=True)
            t_vis = anchor_vis.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                p_vis = decoder_model(x_vis)

            fig = _panel_pred_vs_target(p_vis, t_vis, item=0, max_channels=3)
            wandb.log({"viz/pred_vs_target": wandb.Image(fig), "epoch": epoch + 1})
            plt.close(fig)

            # (선택) epoch 기준 채널별 RMSE
            rmse_epoch = _per_channel_rmse(p_vis, t_vis)
            wandb.log({**{f"rmse_epoch/ch{c}": float(v) for c, v in enumerate(rmse_epoch)},
           "epoch": epoch + 1})

        # --- checkpoint 저장 ---
        # Save base decoder and HF-enhanced full model
        # If wrapped, save the base decoder separately for compatibility
        if isinstance(decoder_model.decoder, FreqRefineDecoder):
            torch.save(decoder_model.decoder.base.state_dict(), "./models/decoder.pth")
            torch.save(decoder_model.decoder.state_dict(), "./models/decoder_hf.pth")
            torch.save(decoder_model.state_dict(), "./models/auto_decoder_hf.pth")
        else:
            torch.save(decoder_model.decoder.state_dict(), "./models/decoder.pth")
            torch.save(decoder_model.state_dict(), "./models/auto_decoder.pth")
