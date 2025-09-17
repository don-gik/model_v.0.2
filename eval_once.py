# eval_once_batched.py
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from module.model import Encoder, RefineBlockDecoder, AxialBlockMLP
from module.dataset import ENSODataset

from tqdm import tqdm

# ----------------- Model wrappers -----------------
class AxialPrediction(nn.Module):
    def __init__(self,
                 T : int = 60,
                 C : int = 4,
                 H : int = 40,
                 W : int = 200,
                 embedDim : int = 64,
                 prediction : int = 10,
                 gradientCheckpointing : bool = True
                 ):
        super().__init__()

        self.encoder = Encoder(
            inChannels = C,
            embedDim = embedDim,
            midDim = 16,
            heads = 4,
            depth = 2,
            window = (5, 10),
            mlpRatio = 2,
            dropout = 0.1
        )

        self.decoder = RefineBlockDecoder(
            channels = embedDim,
            outChannels = C,
            depth = 5,
            drop = 0.1
        )

        self.transformer = AxialBlockMLP(
            C = embedDim,
            T = T,
            H = H // 2,
            W = W // 2,
            heads = 16,
            drop = 0.1
        )

        self.prediction = prediction
        self.gradientCheckpointing = gradientCheckpointing

    def forward(self, x):    # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        z_list = [self.encoder(x[:, t:t+1]).unsqueeze(1) for t in range(T)]
        z = torch.cat(z_list, dim=1)
        z = self.transformer(z)

        out_list = []
        for t in range(T - self.prediction, T):
            zT = z[:, t]
            decoded = self.decoder(zT)
            out_list.append(decoded.unsqueeze(1))
        out = torch.cat(out_list, dim=1)  # [B,prediction,C,H,W]
        return out


class LongTimeAxial(nn.Module):
    def __init__(self,
                 model : AxialPrediction,
                 target : int = 10,
                 T : int = 60):
        super().__init__()
        self.model = model
        self.prediction = target
        self.T = T

    def forward(self, x):
        out_chunks = []
        for _ in range(self.prediction // self.model.prediction):
            y = self.model(x)              # [B, pred, C, H, W]
            x = torch.cat([x, y], dim=1)   # [B, T+pred, C, H, W]
            x = x[:, -self.T:]             # keep last T
            out_chunks.append(y)
        out = torch.cat(out_chunks, dim=1) # [B, T, C, H, W]
        return out[:, -self.prediction:]   # last pred

# ----------------- Utils -----------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_gt_pred_last_panel(pred, target, var_names, out_path):
    """
    pred/target: [1, T, C, H, W]
    """
    ensure_dir(Path(out_path).parent.as_posix())
    pr = pred[0, -1].detach().cpu().numpy()  # [C,H,W]
    gt = target[0, -1].detach().cpu().numpy()
    C = gt.shape[0]
    fig, axes = plt.subplots(2, C, figsize=(3*C, 6), constrained_layout=True)
    for i in range(C):
        vmin = float(min(gt[i].min(), pr[i].min()))
        vmax = float(max(gt[i].max(), pr[i].max()))
        axes[0, i].imshow(gt[i], origin="lower", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"GT {var_names[i]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(pr[i], origin="lower", vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Pred {var_names[i]}")
        axes[1, i].axis("off")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_lead_rmse_curves_mean(rmse_mean, var_names, out_dir):
    """
    rmse_mean: [T, C]
    """
    ensure_dir(out_dir)
    T, C = rmse_mean.shape
    x = np.arange(1, T+1)

    plt.figure(figsize=(8,5), constrained_layout=True)
    for i, name in enumerate(var_names):
        plt.plot(x, rmse_mean[:, i], label=name)
    plt.xlabel("Lead day"); plt.ylabel("RMSE"); plt.title("Lead-wise RMSE (mean over samples)")
    plt.grid(True, linewidth=0.5, alpha=0.5); plt.legend()
    plt.savefig(str(Path(out_dir) / "lead_rmse_all.png"), dpi=150)
    plt.close()

    for i, name in enumerate(var_names):
        plt.figure(figsize=(7,4), constrained_layout=True)
        plt.plot(x, rmse_mean[:, i])
        plt.xlabel("Lead day"); plt.ylabel("RMSE"); plt.title(f"Lead-wise RMSE (mean): {name}")
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.savefig(str(Path(out_dir) / f"lead_rmse_{name}.png"), dpi=150)
        plt.close()

def robust_load(model_wrap, model_core, encoder_path, checkpoint_path, last_target, T_in):
    try:
        enc = torch.load(encoder_path, map_location="cpu")
        model_wrap.model.encoder.load_state_dict(enc)
    except Exception as e:
        print(f"[info] encoder load skipped: {e}")
    try:
        sd = torch.load(checkpoint_path, map_location="cpu")
        try:
            model_core.load_state_dict(sd); print("[info] loaded into AxialPrediction core")
        except:
            try:
                model_wrap.load_state_dict(sd); print("[info] loaded into LongTimeAxial wrapper")
            except:
                try:
                    tmp = LongTimeAxial(model=model_core, target=last_target, T=T_in)
                    tmp.load_state_dict(sd)
                    model_wrap.model = tmp.model
                    print("[info] loaded core from LongTimeAxial checkpoint")
                    del tmp
                except:
                    model_wrap.model.load_state_dict(sd); print("[info] loaded into model_wrap.model")
    except Exception as e:
        print(f"[warn] no full-model checkpoint loaded: {e}")

# ----------------- Main eval -----------------
def main():
    # paths
    encoderDir = "./models/encoder_30days/encoder_long.pth"
    lastCheckpoint = "./models/long_365days/axial_attention_mlp+8.pth"
    dataPath = "./data/enso_avg365.npz"

    # config
    inputDays = 60
    targetDays = 60
    H, W, C = 40, 200, 4
    var_names = ["u10", "v10", "msl", "sst"]
    num_eval = 128
    rng_seed = 1239
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # dataset
    ds = ENSODataset(
        npz_path=dataPath,
        input_days=inputDays,
        target_days=targetDays,
        start_date="1940-01-01"
    )
    N = len(ds)
    np.random.seed(rng_seed)
    eval_indices = np.random.choice(N, size=min(num_eval, N), replace=False)
    subset = Subset(ds, eval_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # model
    core = AxialPrediction(T=inputDays, C=C, H=H, W=W, embedDim=64, prediction=targetDays)
    model = LongTimeAxial(model=core, target=targetDays, T=inputDays).to(device)
    # model = core.to(device)
    model.eval()
    robust_load(model, core, encoderDir, lastCheckpoint, last_target=targetDays, T_in=inputDays)

    # accumulators
    sum_mse = np.zeros((targetDays, C), dtype=np.float64)  # sum over samples of mean_{H,W} MSE
    count = 0

    best_score = float("inf")
    worst_score = -float("inf")
    best_pair = None  # (pred[1,T,C,H,W], target[1,T,C,H,W])
    worst_pair = None

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Eval"):
            xb = xb.to(device)   # [B,T,C,H,W]
            yb = yb.to(device)   # [B,T,C,H,W]

            pred = model(xb)     # [B,targetDays,C,H,W]
            err = pred - yb[:, :pred.size(1)]  # safety slice

            # per-lead per-var MSE averaged over H,W, then sum over batch
            mse_hw = err.pow(2).mean(dim=(3,4))            # [B,Tpred,C]
            sum_mse += mse_hw.sum(dim=0).cpu().numpy()     # [Tpred,C]
            count += xb.size(0)

            # last-lead scalar RMSE per-sample over all vars and pixels
            last_err = err[:, -1]                          # [B,C,H,W]
            rmse_scalar = last_err.pow(2).mean(dim=(1,2,3)).sqrt()  # [B]

            # best in this batch
            b_best = int(torch.argmin(rmse_scalar).item())
            s_best = float(rmse_scalar[b_best].item())
            if s_best < best_score:
                best_score = s_best
                best_pair = (pred[b_best:b_best+1].detach().cpu(),
                             yb[b_best:b_best+1, :pred.size(1)].detach().cpu())

            # worst in this batch
            b_worst = int(torch.argmax(rmse_scalar).item())
            s_worst = float(rmse_scalar[b_worst].item())
            if s_worst > worst_score:
                worst_score = s_worst
                worst_pair = (pred[b_worst:b_worst+1].detach().cpu(),
                              yb[b_worst:b_worst+1, :pred.size(1)].detach().cpu())

    rmse_mean = np.sqrt(sum_mse / max(count, 1))  # [Tpred,C]

    # save outputs
    out_dir = "test"
    ensure_dir(out_dir)
    plot_lead_rmse_curves_mean(rmse_mean, var_names, out_dir=out_dir)
    np.save(str(Path(out_dir) / "lead_rmse_mean.npy"), rmse_mean)

    if best_pair is not None:
        save_gt_pred_last_panel(best_pair[0], best_pair[1], var_names,
                                out_path=str(Path(out_dir) / "gt_vs_pred_last_best.png"))
    if worst_pair is not None:
        save_gt_pred_last_panel(worst_pair[0], worst_pair[1], var_names,
                                out_path=str(Path(out_dir) / "gt_vs_pred_last_worst.png"))

    print("saved:", str(Path(out_dir) / "lead_rmse_all.png"))
    print("saved:", str(Path(out_dir) / "gt_vs_pred_last_best.png"))
    print("saved:", str(Path(out_dir) / "gt_vs_pred_last_worst.png"))
    print("mean RMSE shape:", rmse_mean.shape, "samples:", count)

if __name__ == "__main__":
    main()
