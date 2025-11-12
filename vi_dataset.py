"""
@author: Lochana Marasinghe
@date: 10/17/2025
@description: 
"""

import torch
from torch.utils.data import Dataset

class VIDataset(Dataset):
    def __init__(self, X, y, mask_lengths):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lengths = torch.tensor(mask_lengths, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]