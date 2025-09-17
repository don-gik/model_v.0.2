import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


from module.model import Encoder, RefineBlockDecoder, AxialBlockMLP
from module.dataset import ENSODataset


from tqdm import tqdm
import wandb
import logging

import os
import sys
import random


from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, RandomSampler, DistributedSampler

from piqa import SSIM

import matplotlib
matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt
import numpy as np


from itertools import islice

logging.basicConfig(
    format='[%(asctime)s][%(levelname)s] : %(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

class SSIMLoss(SSIM):
    """
    SSIM Loss for training criterion.
    """
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    RMSE func
    """
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse + eps)


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

        # Encode per day with optional checkpointing to drop activations
        if self.gradientCheckpointing:
            def enc_step(inp): return self.encoder(inp)

            zList = []
            for t in range(T):
                zt = checkpoint(enc_step, x[:, t:t+1], use_reentrant=False).unsqueeze(1)
                zList.append(zt)

            z = torch.cat(zList, dim=1)

            # Axial transformer is the memory hog; checkpoint it
            z = checkpoint(lambda y: self.transformer(y), z, use_reentrant=False)
        else:
            zList = [self.encoder(x[:, t:t+1]).unsqueeze(1) for t in range(T)]
            z = torch.cat(zList, dim=1)

            z = self.transformer(z)


        outList = []
        for t in range(T - self.prediction, T):
            zT = z[:, t]

            if self.gradientCheckpointing:
                decoded = checkpoint(lambda u: self.decoder(u), zT, use_reentrant=False)
            else:
                decoded = self.decoder(zT)

            outList.append(decoded.unsqueeze(1))


        out = torch.cat(outList, dim = 1)

        return out
    

class LongTimeAxial(nn.Module):
    def __init__(self,
                 model : AxialPrediction,
                 target : int = 60,
                 T : int = 60
                 ):
        
        super().__init__()

        self.model = model
        self.prediction = target
        self.T = T
    
    def forward(self, x):
        outList = []

        for _ in range(self.prediction // self.model.prediction):
            y = self.model(x)    # [B, T, C, H, W]
            z = torch.cat([x, y], dim = 1)
            x = z[:, -self.T:]

            outList.append(y)
        
        out = torch.cat(outList, dim = 1)
        
        return out[:, -self.prediction:]





def _wandb_log_lastday_panel(pred: torch.Tensor,
                             target: torch.Tensor,
                             var_names,
                             epoch: int,
                             step: int):
    """
    pred/target: [B, T, C, H, W]
    var_names: ["u10","v10","msl","sst"]
    """
    # random sample from a batch
    b = random.randrange(pred.size(0))
    # use the last 10th day
    gt = target[b, -1].detach().float().cpu().numpy()   # [C, H, W]
    pr = pred[b, -1].detach().float().cpu().numpy()     # [C, H, W]

    C = gt.shape[0]
    fig, axes = plt.subplots(2, C, figsize=(3*C, 6), constrained_layout=True)
    for i in range(C):
        axes[0, i].imshow(gt[i], origin="lower")
        axes[0, i].set_title(f"GT {var_names[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(pr[i], origin="lower")
        axes[1, i].set_title(f"Pred {var_names[i]}")
        axes[1, i].axis("off")

    wandb.log({"viz/lastday_panel": wandb.Image(fig, caption=f"epoch {epoch+1}, sample {b}")},
              step=step)
    plt.close(fig)



def train(encoderDir : str,
          checkpointDir : str,
          lastCheckpoint : str,
          dataPath : str,
          batchesPerEpoch : int = 2048 * 3,
          batchSize : int = 4, 
          epochs : int = 500, 
          lr : float = 1e-4,

          inputDays : int = 60,
          targetDays : int = 10,
          lastTarget : int = 30
          ):
    """
    Training Function for the whole model.
    Model name yet : Axial Attention MLP
    """
    # ---- Kwargs settings for Accelerator setup ----
    ddpKwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision = "bf16",
        gradient_accumulation_steps=4,
        kwargs_handlers = [ddpKwargs]
    )
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- What device? ----
    device = accelerator.device

    logging.info(f"Using device {device}")


    # ---- wandb setup on main process ----
    if accelerator.is_main_process:
        wandb.init(
            project = "Axial Attention MLP",
            name = "long",
            config = {"lr" : lr, "batchSize" : batchSize, "epochs" : epochs}
        )

    # ---- Dataset setup on accelerator ----
    dataset = ENSODataset(
        npz_path = dataPath,
        input_days = inputDays,
        target_days = targetDays,
        start_date = "1940-01-01",
        base_date = "1940-01-01",
        dt_hours = 24
    )

    fullDataset = dataset

    validationN = int(len(fullDataset) * 0.05)
    trainDataset, valDataset = random_split(
        dataset = fullDataset,
        lengths = [len(fullDataset) - validationN, validationN]
    )


    # ---- Load Model Checkpoints ----
    modelCore = AxialPrediction(
        T = inputDays,
        prediction = targetDays
    )
    model = LongTimeAxial(
        model = modelCore,
        target = targetDays,
        T = inputDays
    )

    try:
        # Load encoder checkpoint
        encoderCheckpoint = torch.load(encoderDir, map_location="cpu")
        model.model.encoder.load_state_dict(encoderCheckpoint)
    
    except Exception as e:
        logging.error(f"Error : {e}")
        logging.info("Encoder checkpoint not loaded.")

    try:
        try:
            # Load whole model checkpoint
            modelCheckpoint = torch.load(lastCheckpoint, map_location="cpu")
            model.model.load_state_dict(modelCheckpoint)

            logging.info("Model AxialPred loaded.")
        except:

            try:
                modelCheckpoint = torch.load(lastCheckpoint, map_location="cpu")
                model.load_state_dict(modelCheckpoint)

                logging.info("Model LongAxialPred loaded.")
            
            except:
                try:
                    modelCheckpoint = torch.load(lastCheckpoint, map_location="cpu")

                    loadModel = LongTimeAxial(
                        model = modelCore,
                        target = lastTarget,
                        T = inputDays
                    )

                    loadModel.load_state_dict(modelCheckpoint)

                    model.model = loadModel.model

                    del loadModel
                
                except:

                    modelCheckpoint = torch.load(lastCheckpoint, map_location="cpu")
                    model.model.load_state_dict(modelCheckpoint)

                    logging.info("Model Core has been loaded.")
            
    
    except Exception as e:
        logging.error(f"Error : {e}")
        logging.info("No checkpoint loaded for full model.")

    
    # ---- Setup optimizer, criterion and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr
    )

    C = 4
    criterionL1 = nn.SmoothL1Loss().to(device)

    k = 3.0
    low, high = -k, k
    criterionSsim = SSIMLoss(
        n_channels = C
    ).to(device)
    criterionSsim1 = SSIMLoss(
        n_channels = 1
    ).to(device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",        # 손실 감소를 목표
        factor=0.5,        # lr *= 0.5
        patience=100,        # 3 에폭 개선 없으면 감소
        threshold=1e-2,    # 상대 개선 한계
        threshold_mode="rel",
        cooldown=2,        # 감소 후 1 에폭 대기
        min_lr=lr*0.001,    # lr 하한
        verbose=True
    )


    # ---- Prepare model and optimizer for accelerate ----
    model, optimizer = accelerator.prepare(model, optimizer)

    # Validation Loader prepare
    if accelerator.num_processes > 1:
        train_sampler = DistributedSampler(trainDataset, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(valDataset,   shuffle=False, drop_last=False)

        trainLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=train_sampler,
                                num_workers=2, prefetch_factor=1, pin_memory=True,
                                drop_last=True, persistent_workers=True)
        validationLoader = DataLoader(valDataset, batch_size=batchSize, sampler=val_sampler,
                                    num_workers=2, prefetch_factor=1, pin_memory=True,
                                    drop_last=False, persistent_workers=True)
    else:
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True,
                                num_workers=2, prefetch_factor=1, pin_memory=True,
                                drop_last=True, persistent_workers=True)
        validationLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False,
                                    num_workers=2, prefetch_factor=1, pin_memory=True,
                                    drop_last=False, persistent_workers=True)
    
    validationLoader, trainLoader = accelerator.prepare(validationLoader, trainLoader)

    logging.info("Training fully prepared.")
    logging.info("Training loop started")


    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)


    # ---- Training Loop ----
    for epoch in tqdm(range(epochs)):
        # model training mode

        torch.cuda.reset_peak_memory_stats()
        logging.info(f"peak GB : {torch.cuda.max_memory_allocated()/1e9}")
        logging.info(f"Memory summary : {torch.cuda.memory_summary(device, abbreviated=True)}")
        model.train()
        
        totalLoss = 0.0
        ssimLoss = 0.0
        l1Loss = 0.0

        if hasattr(trainLoader, "sampler") and hasattr(trainLoader.sampler, "set_epoch"):
            trainLoader.sampler.set_epoch(epoch)
        
        training = islice(trainLoader, batchesPerEpoch)

        progressBar = tqdm(
            iterable = training,
            total = batchesPerEpoch,
            position = accelerator.process_index,
            dynamic_ncols = True,
            leave = False,
            disable = False,
            desc = f"Epoch[{epoch}] | R{accelerator.process_index}"
        )

        logging.info(f"Epoch {epoch + 1} started")

        torch.cuda.empty_cache()
        # ---- 1 Epoch Training Loop on freshly sampled dataset ----
        for step, (batchX, batchY) in enumerate(progressBar):
            batchX = batchX[:, :inputDays].to(device)
            batchY = batchY[:, :targetDays].to(device)

            with accelerator.accumulate(model):
                pred = model(batchX)

                def to01(x, k): return ((x.clamp(-k, k) + k) / (2*k))

                lastPred = pred[:, -1]      # [B,C,H,W]
                lastTrue = batchY[:, -1]    # [B,C,H,W]

                lossL1 = criterionL1(pred, batchY)
                lossSsim = criterionSsim(to01(lastPred, k).float(), to01(lastTrue, k).float())

                del lastPred, lastTrue
                loss = lossL1 + lossSsim * 0.2

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        parameters = model.parameters(),
                        max_norm = 0.5
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none = True)

                if accelerator.sync_gradients:
                    scheduler.step(loss)
            
            totalLoss += loss.item()
            ssimLoss += lossSsim.item()
            l1Loss += lossL1.item()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "step_loss" : loss.item(),
                        "step_ssim_loss" : lossSsim.item(),
                        "step_l1_loss" : lossL1.item(),
                        "learning_rate" : scheduler.get_last_lr()[0]
                    },
                    step = epoch * batchesPerEpoch + step
                )
        

        # ---- if an epoch ends ----
        if accelerator.is_main_process:
            totalLoss /= batchesPerEpoch
            ssimLoss  /= batchesPerEpoch
            l1Loss    /= batchesPerEpoch

            wandb.log(
                {
                    "epoch"     : epoch + 1,
                    "loss"      : totalLoss,
                    "ssim_loss" : ssimLoss,
                    "L1_loss"   : l1Loss
                },
                step = (epoch + 1) * batchesPerEpoch
            )


            # Save a Checkpoint
            checkpoint = os.path.join(checkpointDir, f"axial_attention_mlp+{epoch + 1}.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint)
        

        # ---- Validation Loop ----
        model.eval()

        variableLoss = {
            'u10' : {
                'rmse' : 0.0,
                'ssim' : 0.0
            },
            'v10' : {
                'rmse' : 0.0,
                'ssim' : 0.0
            },
            'msl' : {
                'rmse' : 0.0,
                'ssim' : 0.0
            },
            'sst' : {
                'rmse' : 0.0,
                'ssim' : 0.0
            }
        }
        variableNames = ["u10", "v10", "msl", "sst"]

        validationLoss = 0.0

        didLogViz = False

        with torch.no_grad():
            validationProgress = tqdm(
                iterable = validationLoader,
                desc = "Validation",
                position = accelerator.process_index,
                dynamic_ncols = True,
                leave = False,
                disable = False
            )

            for batchX, batchY in validationProgress:
                batchX = batchX.to(device)
                batchY = batchY.to(device)

                pred = model(batchX)

                if accelerator.is_main_process and not didLogViz:
                    _wandb_log_lastday_panel(
                        pred=pred, target=batchY,
                        var_names=variableNames,
                        epoch=epoch,
                        step=(epoch + 1) * batchesPerEpoch
                    )
                    didLogViz = True

                for i, name in enumerate(variableNames):
                    def to01(x, k): return ((x.clamp(-k, k) + k) / (2*k))
                    lossI = rmse_loss(pred[:, -1, i], batchY[:, -1, i])
                    lossIssim = criterionSsim1(to01(pred[:, -1, i].unsqueeze(1), k), to01(batchY[:, -1, i].unsqueeze(1), k))

                    variableLoss[name]['rmse'] += lossI.item()
                    variableLoss[name]['ssim'] += lossIssim.item()

                    validationLoss += lossI.item() + 0.1 * lossIssim.item()
            
        validationLoss /= len(validationLoader)

        for name in variableNames:
            variableLoss[name]['rmse'] /= len(validationLoader)
            variableLoss[name]['ssim'] /= len(validationLoader)

        if accelerator.is_main_process:
            wandb.log(
                {
                    "val_loss" : validationLoss
                },
                step = (epoch + 1) * batchesPerEpoch
            )

            for i, name in enumerate(variableNames):
                wandb.log(
                    {
                        name+"_loss_rmse" : variableLoss[name]['rmse'],
                        name+"_loss_ssim" : variableLoss[name]['ssim']
                    },
                    step = (epoch + 1) * batchesPerEpoch
                )


def main(encoderDir : str,
         checkpointDir : str,
         lastCheckpoint : str,
         dataPath : str,
         batchesPerEpoch : int = 2048 * 3,
         batchSize : int = 4, 
         epochs : int = 500, 
         lr : float = 1e-4,

         inputDays : int = 60,
         targetDays : int = 10,
         lastTarget : int = 30
        ):
    train(
        encoderDir = encoderDir,
        checkpointDir = checkpointDir,
        lastCheckpoint = lastCheckpoint,
        dataPath = dataPath,
        batchesPerEpoch = batchesPerEpoch,
        batchSize = batchSize,
        epochs = epochs,
        lr = lr,

        inputDays = inputDays,
        targetDays = targetDays,
        lastTarget = lastTarget
    )

    return


def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main(
        encoderDir = "./models/encoder_30days/encoder_long.pth",
        checkpointDir = "./models/long_365days",
        lastCheckpoint = "./models/long_365days/axial_attention_mlp+8.pth",
        dataPath = "./data/enso_avg365.npz",
        batchesPerEpoch = 512,
        batchSize = 8,
        epochs = 100,
        lr = 1e-4,

        inputDays = 60,
        targetDays = 60,
        lastTarget = 10
    )