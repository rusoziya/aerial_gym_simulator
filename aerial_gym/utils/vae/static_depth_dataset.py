from __future__ import annotations

import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def load_index(index_csv: str, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(index_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == split:
                rows.append(r)
    return rows


class StaticDepthDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_wh: tuple[int, int] = (240, 135),
        augment: bool = False,
        seed: int = 17,
    ) -> None:
        self.rows = rows
        self.W, self.H = image_wh
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _read_gray(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L").resize((self.W, self.H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]  # [1,H,W]
        return t

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        # t: [1,H,W] in [0,1]
        if self.rng.rand() < 0.15:
            drop_prob = self.rng.uniform(0.01, 0.05)
            mask = torch.from_numpy((self.rng.rand(*t.shape) > drop_prob).astype(np.float32))
            t = t * mask
        if self.rng.rand() < 0.15:
            sigma = self.rng.uniform(0.005, 0.02)
            noise = torch.from_numpy(self.rng.normal(0.0, sigma, size=t.shape).astype(np.float32))
            t = torch.clamp(t + noise, 0.0, 1.0)
        if self.rng.rand() < 0.10:
            gain = self.rng.uniform(0.95, 1.05)
            t = torch.clamp(t * gain, 0.0, 1.0)
        if self.rng.rand() < 0.02:
            t = torch.zeros_like(t)
        if self.rng.rand() < 0.02:
            t = torch.clamp(F.avg_pool2d(t, kernel_size=3, stride=1, padding=1), 0.0, 1.0)
        return t

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.rows[idx]["image_path"]
        t = self._read_gray(path)
        if self.augment:
            t = self._augment(t)
        return t
