import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from module.model.encoder import ASTRAEnhancedEncoder
from module.dataset.ensodataset import ENSOSkipGramDataset
from module.model.decoder import AutoDecoder
import wandb


def spectral_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # pred/true: (B, T, C, H, W)
    # 공간축에 대해 rfftn
    X = torch.fft.rfftn(pred, dim=(-2, -1))
    Y = torch.fft.rfftn(true, dim=(-2, -1))
    return torch.mean(torch.abs(X - Y))
def main():
    # AutoDecoder 불러오기 (방금 정의한 것)
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 250
    wandb.init(project="my-project")
    # 데이터 준비
    npz = np.load("./data/enso_normalized.npz")
    base_data = npz['data'][:, :, :40, :200]
    dataset = ENSOSkipGramDataset(base_data, num_samples=2048)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 구성
    encoder = ASTRAEnhancedEncoder(in_channels=6, embed_dim=256).to(device)
    encoder.load_state_dict(torch.load("./models/astra_encoder.pth", map_location=device))
    decoder_model = AutoDecoder(encoder).to(device)

    # 학습 설정
    optimizer = optim.Adam(decoder_model.decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 학습 루프
    loss_log = []
    try:
        checkpoint = torch.load("./models/auto_decoder.pth", map_location=device)
        decoder_model.load_state_dict(checkpoint)
        print("✅ 이전 체크포인트에서 모델 파라미터 로드됨")
    except FileNotFoundError:
        print("🔁 체크포인트 없음. 새로 학습 시작")
    except:
        print("failure")
    decoder_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, _, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = anchor.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            target = anchor.to(device)          # [B, C, H, W]

            pred = decoder_model(x)             # [B, C, H, W]
            loss = criterion(pred, target) + 0.0 * spectral_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_log.append(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        torch.save(decoder_model.decoder.state_dict(), "./models/decoder.pth")
        torch.save(decoder_model.state_dict(), "./models/auto_decoder.pth")
        wandb.log({"Loss": avg_loss, "epoch": epoch})
        # 중간 결과 저장


    # 저장

