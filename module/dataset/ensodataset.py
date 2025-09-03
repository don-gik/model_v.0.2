import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ENSODataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, input_days=60, target_days=30):
        npz = np.load(npz_path)
        self.data = npz['data'][:, :, :40, :200]
        self.input_days = input_days
        self.target_days = target_days
        self.T = self.data.shape[0]

    def __len__(self):
        return self.T - self.input_days - self.target_days + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_days]      # (input_days, V, H, W)
        y = self.data[idx + self.input_days : idx + self.input_days + self.target_days]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ENSOSkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, base_data, num_samples=16384, pos_radius=3, neg_radius=15, num_negatives=5):
        self.base = base_data  # shape: [T, C, H, W]
        self.T = base_data.shape[0]
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius
        self.num_samples = num_samples
        self.num_negatives = num_negatives

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

