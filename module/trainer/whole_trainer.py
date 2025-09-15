import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, RandomSampler, DistributedSampler

from piqa import SSIM

import matplotlib
matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt
import numpy as np




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
                 prediction : int = 10
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
            depth = 3,
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
    

    def forward(self, x):    # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        
        zList = [self.encoder(x[:, t:t+1]).unsqueeze(1) for t in range(T)]
        z = torch.cat(zList, dim = 1)

        z = self.transformer(z)

        # decoder 
        outList = [self.decoder(z[:, t]).unsqueeze(1) for t in range(T - self.prediction, T)]
        out = torch.cat(outList, dim = 1)

        assert out.shape == (B, self.prediction, C, H, W)

        return out


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
          dataPath : str,
          samplesPerEpoch : int = 2048 * 3,
          batchSize : int = 4, 
          epochs : int = 500, 
          lr : float = 1e-4,

          inputDays : int = 60,
          targetDays : int = 10
          ):
    """
    Training Function for the whole model.
    Model name yet : Axial Attention MLP
    """
    # ---- Kwargs settings for Accelerator setup ----
    ddpKwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers = [ddpKwargs])

    # ---- What device? ----
    device = accelerator.device


    # ---- wandb setup on main process ----
    if accelerator.is_main_process:
        wandb.init(
            project = "Axial Attention MLP Training",
            config = {"lr" : lr, "batchSize" : batchSize, "epochs" : epochs}
        )

    # ---- Dataset setup on accelerator ----
    dataset = ENSODataset(
        npz_path = dataPath,
        input_days = inputDays,
        target_days = targetDays,
        start_data = "1941-01-01",
        base_date = "1941-01-01",
        dt_hours = 24
    )

    fullDataset = dataset

    validationN = int(len(fullDataset) * 0.05)
    trainDataset, valDataset = random_split(
        dataset = fullDataset,
        lengths = [len(fullDataset) - validationN, validationN]
    )

    sampler = DistributedSampler(
        dataset = dataset,
        shuffle = True
    )  if accelerator.num_processes > 1 else None


    # ---- Load Model Checkpoints ----
    model = AxialPrediction(
        T = inputDays,
        prediction = targetDays
    )

    try:
        # Load encoder checkpoint
        encoderCheckpoint = torch.load(encoderDir, map_location="cpu")
        model.encoder.load_state_dict(encoderCheckpoint)
    
    except Exception as e:
        pass

    try:
        # Load whole model checkpoint
        modelCheckpoint = torch.load(checkpointDir, map_location="cpu")
        model.load_state_dict(modelCheckpoint)
    
    except Exception as e:
        pass

    
    # ---- Setup optimizer, criterion and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr
    )

    criterionL1 = nn.SmoothL1Loss()
    criterionSsim = SSIMLoss()

    scheduler = OneCycleLR(
        optimizer = optimizer,
        max_lr = 1e-3,
        steps_per_epoch = samplesPerEpoch // batchSize,
        epochs = epochs,
        pct_start = 0.44,
        anneal_strategy = 'cos'
    )


    # ---- Prepare model and optimizer for accelerate ----
    model, optimizer = accelerator.prepare(model, optimizer)

    # Validation Loader prepare
    validationLoader = DataLoader(
        dataset = valDataset,
        batch_size = 4,
        shuffle = False,
        num_workers = 0,
        pin_memory = False
    )
    validationLoader = accelerator.prepare(validationLoader)



    # ---- Training Loop ----
    for epoch in tqdm(range(epochs)):
        # model training mode
        model.train()
        
        totalLoss = 0.0
        ssimLoss = 0.0
        l1Loss = 0.0

        # Sampling the Train Dataloader
        generator = torch.Generator().manual_seed(random.randint())

        sampler = RandomSampler(
            data_source = trainDataset,
            replacement = False,
            num_samples = samplesPerEpoch,
            generator = generator
        )

        trainLoader = DataLoader(
            dataset = trainDataset,
            sampler = sampler,
            batch_size = batchSize,
            num_workers = 2,
            pin_memory = False,
            drop_last = True
        )

        # Prepare train loader
        trainLoader = accelerator.prepare(trainLoader)

        progressBar = tqdm(
            iterable = trainLoader,
            position = accelerator.process_index,
            dynamic_ncols = True,
            leave = False,
            disable = False,
            desc = f"Epoch[{epoch}] | R{accelerator.process_index}"
        )


        # ---- 1 Epoch Training Loop on freshly sampled dataset ----
        for step, (batchX, batchY) in enumerate(progressBar):
            batchX = batchX[:, :inputDays].to(device)
            batchY = batchY[:, :targetDays].to(device)

            with accelerator.accumulate(model):
                pred = model(batchX)

                lossL1 = criterionL1(pred, batchY)
                lossSsim = (1.0 - criterionSsim(pred, batchY))
                loss = lossL1 + lossSsim

                optimizer.zero_grad()
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(
                    parameters = model.parameters(),
                    max_norm = 0.5
                )

                optimizer.step()
                scheduler.step()
            
            totalLoss += loss.item()
            ssimLoss += lossSsim.item()
            l1Loss += lossL1.item()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "step_loss" : loss.item(),
                        "step_ssim_loss" : lossSsim.item(),
                        "step_l1_loss" : lossL1.item()
                    },
                    step = epoch * len(trainLoader) + step
                )
        

        # ---- if an epoch ends ----
        if accelerator.is_main_process:
            totalLoss /= len(trainLoader)
            ssimLoss  /= len(trainLoader)
            l1Loss    /= len(trainLoader)

            wandb.log(
                {
                    "epoch"     : epoch + 1,
                    "loss"      : totalLoss,
                    "ssim_loss" : ssimLoss,
                    "L1_loss"   : l1Loss
                },
                step = (epoch + 1) * len(trainLoader)
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

                if accelerator.is_main_process and not did_log_viz:
                    _wandb_log_lastday_panel(
                        pred=pred, target=batchY,
                        var_names=variableNames,
                        epoch=epoch,
                        step=(epoch + 1) * len(trainLoader)
                    )
                    did_log_viz = True

                for i, name in enumerate(variableNames):
                    lossI = rmse_loss(pred[:, :, i], batchY[:, :, i])
                    lossIssim = 1.0 - SSIMLoss(pred[:, :, i], batchY[:, :, i])

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
                step = (epoch + 1) * len(validationLoader)
            )

            for i, name in enumerate(variableNames):
                wandb.log(
                    {
                        name+"_loss_rmse" : variableLoss[name]['rmse'],
                        name+"_loss_ssim" : variableLoss[name]['ssim']
                    },
                    step = (epoch + 1) * len(trainLoader)
                )


def main(encoderDir : str,
         checkpointDir : str,
         dataPath : str,
         samplesPerEpoch : int = 2048 * 3,
         batchSize : int = 4, 
         epochs : int = 500, 
         lr : float = 1e-4,

         inputDays : int = 60,
         targetDays : int = 10
        ):
    train(
        encoderDir = encoderDir,
        checkpointDir = checkpointDir,
        dataPath = dataPath,
        samplesPerEpoch = samplesPerEpoch,
        batchSize = batchSize,
        epochs = epochs,
        lr = lr,

        inputDays = inputDays,
        targetDays = targetDays
    )

    return


if __name__ == "__main__":
    main(
        encoderDir = "./models/encoder/encoder_best.pth",
        checkpointDir = "./models/whole",
        dataPath = "./data/enso_normalized.npz",
        samplesPerEpoch = 2048 * 3,
        batchSize = 8,
        epochs = 500,
        lr = 1e-4,

        inputDays = 60,
        targetDays = 10
    )