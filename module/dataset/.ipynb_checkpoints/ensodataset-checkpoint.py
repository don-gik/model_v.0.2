import numpy as np
import torch
from torch.utils.data import Dataset
import random

# --- 공통: 날짜 → 시작 인덱스 유틸 ---
def _index_from_date(start_date: str,
                     base_date: str = "1941-01-01",
                     dt_hours: int = 24) -> int:
    """
    start_date로 잘라내기 위한 시간축 시작 인덱스 계산.
    npz 데이터의 시간 해상도가 dt_hours(기본 일단위=24시간)라고 가정.
    """
    # 경과 시간을 '시간' 단위로 계산 후, 샘플 간격(dt_hours)로 나눠 인덱스 변환
    delta_hours = (np.datetime64(start_date) - np.datetime64(base_date)) / np.timedelta64(1, 'h')
    idx = int(delta_hours // dt_hours)
    return max(idx, 0)

class ENSODataset(Dataset):
    def __init__(self,
                 npz_path,
                 input_days=60,
                 target_days=30,
                 start_date: str = "1978-01-01",     # <- 여기만 바꾸면 시작 연/월/일 필터링
                 base_date: str = "1941-01-01",
                 dt_hours: int = 24):                 # 일단위면 24, 6시간 간격이면 6 등으로 조정
        npz = np.load(npz_path)
        data = npz['data'][:, :, :40, :200]          # (T, C, H, W)

        # 시작 날짜 → 인덱스 변환 후, 앞부분 컷
        if start_date is not None:
            s = _index_from_date(start_date, base_date, dt_hours)
            if s >= data.shape[0]:
                raise ValueError(f"start_date={start_date}가 데이터 범위를 벗어남 (T={data.shape[0]})")
            data = data[s:]

        self.data = data
        self.input_days = input_days
        self.target_days = target_days
        self.T = self.data.shape[0]

        # 시퀀스 길이 확인 (음수 방지)
        usable = self.T - self.input_days - self.target_days + 1
        if usable <= 0:
            raise ValueError(
                f"시퀀스가 부족함: T={self.T}, input_days={input_days}, target_days={target_days}"
            )

    def __len__(self):
        return self.T - self.input_days - self.target_days + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_days]                         # (input_days, V, H, W)
        y = self.data[idx + self.input_days : idx + self.input_days + self.target_days]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ENSOSkipGramDataset(Dataset):
    def __init__(self,
                 base_data,                    # shape: [T, C, H, W]
                 num_samples=16384,
                 pos_radius=3,
                 neg_radius=15,
                 num_negatives=5,
                 start_date: str = "1978-01-01",    # 동일하게 필터 지원
                 base_date: str = "1941-01-01",
                 dt_hours: int = 24):
        if start_date is not None:
            s = _index_from_date(start_date, base_date, dt_hours)
            if s >= base_data.shape[0]:
                raise ValueError(f"start_date={start_date}가 데이터 범위를 벗어남 (T={base_data.shape[0]})")
            base_data = base_data[s:]

        self.base = base_data
        self.T = base_data.shape[0]
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius
        self.num_samples = num_samples
        self.num_negatives = num_negatives

        if self.T < (2 * max(pos_radius, neg_radius) + 1):
            raise ValueError(f"샘플링에 필요한 최소 길이 부족: T={self.T}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        anchor_t = random.randint(self.pos_radius, self.T - self.pos_radius - 1)
        anchor = self.base[anchor_t]

        pos_offsets = [i for i in range(-self.pos_radius, self.pos_radius+1) if i != 0]
        pos_t = anchor_t + random.choice(pos_offsets)
        positive = self.base[pos_t]

        negs = []
        while len(negs) < self.num_negatives:
            neg_t = random.randint(0, self.T - 1)
            if abs(neg_t - anchor_t) > self.neg_radius:
                negs.append(self.base[neg_t])

        return (
            torch.tensor(anchor, dtype=torch.float32),     # [C, H, W]
            torch.tensor(positive, dtype=torch.float32),   # [C, H, W]
            torch.stack([torch.tensor(n, dtype=torch.float32) for n in negs])  # [N, C, H, W]
        )
