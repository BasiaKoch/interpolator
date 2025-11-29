from __future__ import annotations
import numpy as np, torch
from torch import nn
from dataclasses import dataclass

@dataclass
class MLPConfig:
    hidden: tuple[int,...] = (128,64,32)
    lr: float = 1e-3
    max_epochs: int = 150
    batch_size: int = 256
    patience: int = 15
    seed: int = 42

class MLPRegressor:
    def __init__(self, cfg: MLPConfig = MLPConfig()):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        layers, in_dim = [], 5
        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.device = torch.device("cpu")
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.loss = nn.MSELoss()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=self.device)
        ds = torch.utils.data.TensorDataset(X,y)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        best, bad = np.inf, 0
        for epoch in range(self.cfg.max_epochs):
            self.net.train()
            for xb, yb in dl:
                self.opt.zero_grad()
                l = self.loss(self.net(xb), yb); l.backward(); self.opt.step()
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    val = self.loss(self.net(torch.tensor(X_val, dtype=torch.float32)),
                                    torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)).item()
                if val < best - 1e-6: best, bad = val, 0
                else: bad += 1
                if bad >= self.cfg.patience: break
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit first"
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32)).cpu().numpy().ravel()
        return out

