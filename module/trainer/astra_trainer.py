import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from module.model.encoder import Encoder
from module.model.decoder import AutoDecoder
from module.model.transformer import ASTRA_Block
from module.dataset.ensodataset import ENSODataset
from tqdm import tqdm
import wandb
import os
import logging
import sys
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, random_split

from module.model.decoder.decoder import Decoder
from module.model.decoder.hf_refine_decoder import FreqRefineDecoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# --- Setup logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.__stdout__)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def spectral_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # pred/true: (B, T, C, H, W)
    # 공간축에 대해 rfftn
    X = torch.fft.rfftn(pred, dim=(-2, -1))
    Y = torch.fft.rfftn(true, dim=(-2, -1))
    return torch.mean(torch.abs(X - Y))

import torch
import torch.nn.functional as F
from torch import nn

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        # img1, img2: (B, C, T, H, W) → SSIM은 2D에 적용, 시간축으로 평균
        B, C, T, H, W = img1.shape
        ssim_total = 0.0
        for t in range(T):
            x = img1[:, :, t, :, :]
            y = img2[:, :, t, :, :]
            ssim = self._ssim_2d(x, y)
            ssim_total += (1 - ssim)  # SSIM은 유사도 → 1-SSIM이 loss
        return ssim_total / T

    def _ssim_2d(self, x, y, C1=0.01**2, C2=0.03**2):
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size // 2)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.avg_pool2d(x * x, self.window_size, stride=1, padding=self.window_size // 2) - mu_x2
        sigma_y2 = F.avg_pool2d(y * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_y2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim = numerator / (denominator + 1e-6)
        if self.size_average:
            return ssim.mean()
        else:
            return ssim.view(x.size(0), -1).mean(1)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    RMSE 손실 함수 (함수형)
    :param pred: 예측값, shape [B, ...]
    :param target: 실제값, shape [B, ...]
    :param eps: 0 나누기 방지용 작은 상수
    :return: 스칼라 RMSE
    """
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse + eps)


# --- Define ASTRA AutoDecoder ---
class ASTRAAutoDecoder(nn.Module):
    def __init__(self, encoder, decoder, dim=256, T=60, H=20, W=100):
        super().__init__()
        self.encoder = encoder
        self.transformer = ASTRA_Block(dim, T, H, W)
        self.decoder = decoder

        # for p in self.encoder.parameters():
        #    p.requires_grad = False
        # for p in self.decoder.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape
        # encode per timestep
        z_list = [self.encoder(x[:, t:t+1]).unsqueeze(1) for t in range(T)]
        z = torch.cat(z_list, dim=1)
        z = self.transformer(z)

        # decode only last 30 steps
        out_list = [self.decoder(z[:, t]).unsqueeze(1) for t in range(T - 1, T)]
        out = torch.cat(out_list, dim=1)
        # return in (B, C, T, H, W) order
        return out.permute(0, 2, 1, 3, 4)

# --- Training function ---
def train_astra_autodecoder(encoder_ckpt, decoder_ckpt, data_path, save_path,
                            batch_size=2, epochs=500, lr=1e-4):
    from torch.utils.data import DataLoader, RandomSampler
    # --- Accelerator setup with DDP kwargs ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # --- wandb only on main process ---
    if accelerator.is_main_process:
        wandb.init(project="ASTRA-AutoDecoder", config={"lr": lr, "batch_size": batch_size})

    # --- Dataset & DistributedSampler ---
    dataset = ENSODataset(data_path, input_days=60, target_days=1)
    full_ds = dataset
    n_val   = int(len(full_ds) * 0.01)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    sampler = DistributedSampler(dataset, shuffle=True) if accelerator.num_processes > 1 else None

    # --- Load checkpoints ---
    encoder_sd = torch.load(encoder_ckpt, map_location="cpu")
    decoder_sd = torch.load(decoder_ckpt, map_location="cpu")

    # --- Build models ---
    encoder = Encoder(in_channels=4, embed_dim=128, mid_dim=32)
    encoder.load_state_dict(encoder_sd)

    # Note: AutoDecoder takes encoder as argument but we only need its decoder part here
    base_dec = Decoder(in_channels=encoder.out_dim, out_channels=4, hidden=128, num_res=2, dropout=0.05, use_deconv=True)
    dec = FreqRefineDecoder(base_dec, in_channels=encoder.out_dim, out_channels=4, cond_z=True, refine_width=96, refine_depth=3, dropout=0.05)
    try:
        dec.load_state_dict(decoder_sd)
    except:
        logger.info(f"Decoder loading failed.")
    decoder_model = dec.decoder

    model = ASTRAAutoDecoder(encoder, decoder_model)

    try:
        ckpt_path = "./models/astra_autodecoder_endecoder+.pth"  # 불러올 .pth 파일 경로
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except:
        logger.warning(f"Failed to load checkpoint from {ckpt_path}. Starting from scratch.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5
    )
    criterion = nn.SmoothL1Loss()  # HuberLoss
    criterion_ssim = SSIMLoss(window_size=11, size_average=True)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,                       # peak LR: 실험적으로 정하세요
        steps_per_epoch=3*512,
        epochs=epochs,
        pct_start=0.44,                     # warm-up 비율
        anneal_strategy="cos"
    )

    # --- Dataloader with sampler ---
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)

    # --- Prepare for distributed ---
    # ... (생략된 코드 동일) ...

    # --- Prepare model and optimizer ---
    model, optimizer = accelerator.prepare(model, optimizer)

    # ❗️ val_loader도 prepare해야 DDP가 올바르게 작동함
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=False
    )
    val_loader = accelerator.prepare(val_loader)

    logger.info("✅ Model, optimizer, and dataloader prepared.")

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        g = torch.Generator().manual_seed(65373 + epoch)

        # --- 랜덤 샘플링 매 epoch 생성
        sampler = RandomSampler(
            train_ds, replacement=False,
            num_samples=3 * 1024, generator=g
        )
        loader = DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader = accelerator.prepare_data_loader(loader)

        progress_bar = tqdm(loader,
                            disable=not accelerator.is_main_process,
                            desc=f"Epoch {epoch+1}",
                            leave=True,
                            file=sys.stdout)

        for step, (xb, yb) in enumerate(progress_bar):
            xb = xb[:, :60].to(device)
            yb = yb.permute(0, 2, 1, 3, 4).to(device)

            with accelerator.accumulate(model):
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            if accelerator.is_main_process:
                wandb.log({"step_loss": loss.item()}, step=epoch * len(loader) + step)

        if accelerator.is_main_process:
            total_loss /= len(loader)
            logger.info(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "loss": total_loss}, step=(epoch + 1) * len(loader))

            # save checkpoint
            ckpt_path = os.path.join(save_path, f"astra_autodecoder_endecoder+.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)

        # --- Validation ---
        model.eval()
        var_loss_sums = {'u10' : 0.0, 'v10' : 0.0, 'msl' : 0.0, 'sst' : 0.0}
        var_names = ["u10", "v10", "msl", "sst"]
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="  Valid", leave=False, disable=not accelerator.is_main_process):
                xb = xb.to(device)
                # yb: 원래 (B, T, C, H, W) → (B, C, T, H, W)
                yb = yb.permute(0, 2, 1, 3, 4).to(device)

                pred = model(xb)  # shape [B, C, T, H, W]
                
                # 변수별로 RMSE 계산
                for i, name in enumerate(var_names):
                    # pred[:, i] 와 yb[:, i] 의 shape은 [B, T, H, W]
                    loss_i = rmse_loss(pred[:, i], yb[:, i])
                    var_loss_sums[name] += loss_i.item()
                    val_loss += loss_i.item()
                

        val_loss /= len(val_loader)

        for name in var_names:
            var_loss_sums[name] /= len(val_loader)
        if accelerator.is_main_process:
            logger.info(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")
            wandb.log({"val_loss": val_loss}, step=(epoch + 1) * len(loader))
            for i, name in enumerate(var_names):
                wandb.log({name + "_loss": var_loss_sums[name]}, step=(epoch+1) * len(loader))

            with torch.no_grad():
                vis_input = xb[0:1].detach().cpu().to(device)  # [1, T, C, H, W]
                vis_target = yb[0:1].detach().cpu().numpy()    # [1, C, T, H, W]
                vis_pred = accelerator.unwrap_model(model)(vis_input).detach().cpu().numpy()  # [1, C, T, H, W]

                # 변수 및 시간 선택
                var_idx = 0
                t = 0
                true_map = vis_target[0, var_idx, t]
                pred_map = vis_pred[0, var_idx, t]
                diff_map = pred_map - true_map

                wandb.log({
                    f"true_map/u10 epoch{epoch+1}": wandb.Image(true_map, caption=f"True - epoch {epoch+1}"),
                    f"pred_map/u10 epoch{epoch+1}": wandb.Image(pred_map, caption=f"Predicted - epoch {epoch+1}"),
                    f"diff_map/u10 epoch{epoch+1}": wandb.Image(diff_map, caption=f"Diff - epoch {epoch+1}")
                }, step=(epoch + 1) * len(loader))

                var_idx = 1
                t = 0
                true_map = vis_target[0, var_idx, t]
                pred_map = vis_pred[0, var_idx, t]
                diff_map = pred_map - true_map

                wandb.log({
                    f"true_map/v10 epoch{epoch+1}": wandb.Image(true_map, caption=f"True - epoch {epoch+1}"),
                    f"pred_map/v10 epoch{epoch+1}": wandb.Image(pred_map, caption=f"Predicted - epoch {epoch+1}"),
                    f"diff_map/v10 epoch{epoch+1}": wandb.Image(diff_map, caption=f"Diff - epoch {epoch+1}")
                }, step=(epoch + 1) * len(loader))

                var_idx = 2
                t = 0
                true_map = vis_target[0, var_idx, t]
                pred_map = vis_pred[0, var_idx, t]
                diff_map = pred_map - true_map

                wandb.log({
                    f"true_map/msl epoch{epoch+1}": wandb.Image(true_map, caption=f"True - epoch {epoch+1}"),
                    f"pred_map/msl epoch{epoch+1}": wandb.Image(pred_map, caption=f"Predicted - epoch {epoch+1}"),
                    f"diff_map/msl epoch{epoch+1}": wandb.Image(diff_map, caption=f"Diff - epoch {epoch+1}")
                }, step=(epoch + 1) * len(loader))

                var_idx = 3
                t = 0
                true_map = vis_target[0, var_idx, t]
                pred_map = vis_pred[0, var_idx, t]
                diff_map = pred_map - true_map

                wandb.log({
                    f"true_map/sst epoch{epoch+1}": wandb.Image(true_map, caption=f"True - epoch {epoch+1}"),
                    f"pred_map/sst epoch{epoch+1}": wandb.Image(pred_map, caption=f"Predicted - epoch {epoch+1}"),
                    f"diff_map/sst epoch{epoch+1}": wandb.Image(diff_map, caption=f"Diff - epoch {epoch+1}")
                }, step=(epoch + 1) * len(loader))

        accelerator.wait_for_everyone()

import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_numpy(y_true, y_pred, epoch, var_idx=0, save_path="epoch_vis.png"):
    """
    y_true, y_pred: (C, T, H, W)
    """
    t = 0  # 예측된 1일만 시각화
    true_map = y_true[var_idx, t]
    pred_map = y_pred[var_idx, t]
    diff_map = pred_map - true_map

    vmin = min(true_map.min(), pred_map.min())
    vmax = max(true_map.max(), pred_map.max())
    max_diff = np.abs(diff_map).max()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(true_map, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"[Epoch {epoch}] True")
    axs[1].imshow(pred_map, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[1].set_title("Predicted")
    axs[2].imshow(diff_map, cmap='bwr', vmin=-max_diff, vmax=max_diff)
    axs[2].set_title("Difference")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    train_astra_autodecoder(
        encoder_ckpt='./models/astra_encoder.pth',
        decoder_ckpt='./models/auto_decoder.pth',
        data_path='./data/enso_normalized.npz',
        save_path='./models',
        batch_size=8,
        epochs=500,
        lr=1e-4
    )
