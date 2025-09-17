# eval_refs_acc.py
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
matplotlib = None  # no plotting

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
                 prediction : int = 60,
                 gradientCheckpointing : bool = True):
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


# ----------------- Utils -----------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def robust_load(model_wrap, model_core, encoder_path, checkpoint_path, last_target, T_in):
    try:
        enc = torch.load(encoder_path, map_location="cpu")
        model_wrap.encoder.load_state_dict(enc, strict=False)
    except Exception as e:
        print(f"[info] encoder load skipped: {e}")
    try:
        sd = torch.load(checkpoint_path, map_location="cpu")
        try:
            model_core.load_state_dict(sd); print("[info] loaded into AxialPrediction core")
        except Exception:
            try:
                model_wrap.load_state_dict(sd); print("[info] loaded full wrapper state")
            except Exception as e:
                print(f"[warn] could not load checkpoint strictly: {e}")
    except Exception as e:
        print(f"[warn] no full-model checkpoint loaded: {e}")

def try_lat_weights_from_dataset(ds, H, W):
    cand = None
    for k in ("lats_deg", "lats", "lat", "latitude"):
        if hasattr(ds, k):
            cand = getattr(ds, k)
            break
        if isinstance(getattr(ds, "__dict__", {}), dict) and k in ds.__dict__:
            cand = ds.__dict__[k]
            break
    if cand is None:
        print("[info] no latitude vector found in dataset → use uniform weights")
        w = torch.ones(H, W, dtype=torch.float32)
        return w, False
    lat = torch.tensor(np.asarray(cand), dtype=torch.float32)  # [H] expected
    w = torch.cos(torch.deg2rad(lat)).clamp(min=0)[:, None].repeat(1, W)  # [H,W]
    return w, True

def rmse_skill(model_rmse, ref_rmse, eps=1e-12):
    return 1.0 - (model_rmse**2) / (ref_rmse**2 + eps)


# ----------------- Main eval -----------------
def main():
    # paths
    encoderDir = "./models/encoder_30days/encoder_long.pth"
    lastCheckpoint = "./models/whole_365days_decoder+/axial_attention_mlp+10.pth"
    dataPath = "./data/enso_avg365.npz"

    # config
    inputDays = 60
    targetDays = 10
    H, W, C = 40, 200, 4
    var_names = ["u10", "v10", "msl", "sst"]
    # optional std to convert RMSE_z → physical units
    sigma_per_var = {"sst": 1.4}  # fill others if known
    num_eval = 128
    rng_seed = 1239
    batch_size = 4
    out_dir = "test_metrics"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # dataset
    ds = ENSODataset(
        npz_path=dataPath,
        input_days=inputDays,
        target_days=targetDays,
        start_date="1940-01-01"
    )
    Nall = len(ds)
    np.random.seed(rng_seed)
    eval_indices = np.random.choice(Nall, size=min(num_eval, Nall), replace=False)
    subset = Subset(ds, eval_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # lat-area weights
    w2d_cpu, has_lat = try_lat_weights_from_dataset(ds, H, W)
    wsum = float(w2d_cpu.sum().item())

    # model
    core = AxialPrediction(T=inputDays, C=C, H=H, W=W, embedDim=64, prediction=targetDays)
    model = core.to(device).eval()
    robust_load(model, core, encoderDir, lastCheckpoint, last_target=targetDays, T_in=inputDays)

    # accumulators (CPU, float64 for stability)
    L = targetDays
    sum_mse_model = torch.zeros(C, L, dtype=torch.float64)
    sum_mse_clim  = torch.zeros(C, L, dtype=torch.float64)
    sum_mse_pers  = torch.zeros(C, L, dtype=torch.float64)

    sum_y  = torch.zeros(C, L, H, W, dtype=torch.float64)
    sum_p  = torch.zeros(C, L, H, W, dtype=torch.float64)
    sum_yy = torch.zeros(C, L, H, W, dtype=torch.float64)
    sum_pp = torch.zeros(C, L, H, W, dtype=torch.float64)
    sum_yp = torch.zeros(C, L, H, W, dtype=torch.float64)

    count = 0

    with torch.no_grad():
        w2d = w2d_cpu.to(device)

        for xb, yb in tqdm(loader, desc="Eval"):
            xb = xb.to(device)                 # [B,T,C,H,W]
            yb = yb.to(device)[:, :L]         # [B,L,C,H,W]
            B = xb.size(0)

            pred = model(xb)[:, :L]           # [B,L,C,H,W]

            for v in range(C):
                y = yb[:, :, v]               # [B,L,H,W]
                p = pred[:, :, v]             # [B,L,H,W]
                y0 = xb[:, -1, v]             # [B,H,W]  persistence origin

                # --- RMSE model ---
                err2 = ((y - p).pow(2) * w2d).sum((-1, -2)) / wsum  # [B,L]
                sum_mse_model[v] += err2.sum(0).double().cpu()

                # --- RMSE climatology (zero for standardized anomalies) ---
                e2c = ((y).pow(2) * w2d).sum((-1, -2)) / wsum       # [B,L]
                sum_mse_clim[v] += e2c.sum(0).double().cpu()

                # --- RMSE persistence ---
                e2p = ((y - y0[:, None]).pow(2) * w2d).sum((-1, -2)) / wsum  # [B,L]
                sum_mse_pers[v] += e2p.sum(0).double().cpu()

                # --- ACC streaming stats ---
                # sums across batch dimension
                y_cpu = y.double().cpu()
                p_cpu = p.double().cpu()
                sum_y[v]  += y_cpu.sum(0)
                sum_p[v]  += p_cpu.sum(0)
                sum_yy[v] += (y_cpu * y_cpu).sum(0)
                sum_pp[v] += (p_cpu * p_cpu).sum(0)
                sum_yp[v] += (y_cpu * p_cpu).sum(0)

            count += B

    # finalize metrics
    eps = 1e-12
    rmse_model = torch.sqrt(sum_mse_model / max(count, 1.0))  # [C,L]
    rmse_clim  = torch.sqrt(sum_mse_clim  / max(count, 1.0))
    rmse_pers  = torch.sqrt(sum_mse_pers  / max(count, 1.0))

    skill_vs_clim = rmse_skill(rmse_model, rmse_clim, eps)    # [C,L]
    skill_vs_pers = rmse_skill(rmse_model, rmse_pers, eps)

    # ACC per var, lead
    ACC = torch.zeros(C, L, dtype=torch.float64)
    w2d_cpu64 = w2d_cpu.double()
    wsum64 = w2d_cpu64.sum()

    for v in range(C):
        N = float(count)
        num = (sum_yp[v] - (sum_y[v] * sum_p[v]) / N)                         # [L,H,W]
        den = torch.sqrt((sum_yy[v] - sum_y[v]**2 / N) *
                         (sum_pp[v] - sum_p[v]**2 / N) + eps)                 # [L,H,W]
        rmap = torch.clamp(num / den, -1.0, 1.0)                              # [L,H,W]
        # area-weighted average to scalar ACC per lead
        ACC[v] = ((rmap * w2d_cpu64).sum((-1, -2)) / wsum64)

    # Optional: convert RMSE_z → physical units using sigma_per_var
    rmse_model_phys = rmse_model.clone()
    rmse_clim_phys  = rmse_clim.clone()
    rmse_pers_phys  = rmse_pers.clone()
    for i, name in enumerate(var_names):
        if name in sigma_per_var:
            s = float(sigma_per_var[name])
            rmse_model_phys[i] *= s
            rmse_clim_phys[i]  *= s
            rmse_pers_phys[i]  *= s

    # save
    ensure_dir(out_dir)
    np.save(Path(out_dir, "rmse_model_z.npy"), rmse_model.numpy())        # [C,L]
    np.save(Path(out_dir, "rmse_climatology_z.npy"), rmse_clim.numpy())
    np.save(Path(out_dir, "rmse_persistence_z.npy"), rmse_pers.numpy())
    np.save(Path(out_dir, "skill_vs_clim.npy"), skill_vs_clim.numpy())
    np.save(Path(out_dir, "skill_vs_pers.npy"), skill_vs_pers.numpy())
    np.save(Path(out_dir, "acc.npy"), ACC.numpy())
    np.save(Path(out_dir, "rmse_model_phys.npy"), rmse_model_phys.numpy())
    np.save(Path(out_dir, "rmse_climatology_phys.npy"), rmse_clim_phys.numpy())
    np.save(Path(out_dir, "rmse_persistence_phys.npy"), rmse_pers_phys.numpy())

    # brief print
    print(f"samples: {count}")
    for i, name in enumerate(var_names):
        r1 = float(rmse_model[i,0])
        rL = float(rmse_model[i,-1])
        a1 = float(ACC[i,0])
        aL = float(ACC[i,-1])
        sC1 = float(skill_vs_clim[i,0])
        sCL = float(skill_vs_clim[i,-1])
        sP1 = float(skill_vs_pers[i,0])
        sPL = float(skill_vs_pers[i,-1])
        print(f"[{name}] lead1/lead{L}  RMSE_z: {r1:.4f} / {rL:.4f}  "
            f"ACC: {a1:.3f} / {aL:.3f}  "
            f"Skill vs clim: {sC1:.3f} / {sCL:.3f}  "
            f"Skill vs pers: {sP1:.3f} / {sPL:.3f}")

if __name__ == "__main__":
    main()
