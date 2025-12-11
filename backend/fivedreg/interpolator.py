import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ----------------------------------------------------------------------
# Normalisation utilities
# ----------------------------------------------------------------------


@dataclass
class NormStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def compute_norm_stats(X: np.ndarray, y: np.ndarray) -> NormStats:
    """
    Compute simple mean/std normalisation statistics for inputs and target.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-8)

    return NormStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def apply_x_norm(X: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Normalise input features with stored stats.
    """
    return (X - stats.x_mean) / stats.x_std


def apply_y_norm(y: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Normalise target with stored stats.
    """
    return (y - stats.y_mean) / stats.y_std


def invert_y_norm(yn: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Invert target normalisation back to original units.
    """
    return yn * stats.y_std + stats.y_mean


# ----------------------------------------------------------------------
# Simple MLP
# ----------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 64,
        depth: int = 3,
        out_dim: int = 1,
        act=nn.SiLU,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ----------------------------------------------------------------------
# Training loop with early stopping
# ----------------------------------------------------------------------


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    max_epochs: int = 200,
    lr: float = 5e-3,
    weight_decay: float = 1e-6,
    patience: int = 20,
) -> Tuple[MLP, NormStats, Tuple[float, float, float, float]]:
    print("=" * 60)
    print("🚀 USING NEW INTERPOLATOR.PY CODE")
    print("=" * 60)
    stats = compute_norm_stats(X, y)
    Xn = apply_x_norm(X, stats).astype(np.float32)
    yn = apply_y_norm(y, stats).astype(np.float32)

    Xn_t = torch.from_numpy(Xn)
    yn_t = torch.from_numpy(yn).unsqueeze(1)

    ds = TensorDataset(Xn_t, yn_t)

    n_total = len(ds)
    # Use percentage-based splits, with minimum reasonable sizes
    n_val = max(int(0.15 * n_total), min(200, int(0.15 * n_total)))
    n_test = max(int(0.15 * n_total), min(200, int(0.15 * n_total)))
    n_train = n_total - n_val - n_test

    # Ensure we have at least some samples in each split
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        # For very small datasets, use simple percentage splits
        n_val = max(1, int(0.15 * n_total))
        n_test = max(1, int(0.15 * n_total))
        n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(123),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        sched.step()

        # Validation
        model.eval()
        mse = 0.0
        n = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mse += nn.functional.mse_loss(
                    pred, yb, reduction="sum"
                ).item()
                n += xb.size(0)
                val_preds.append(pred.cpu())
                val_targets.append(yb.cpu())
        val_loss = mse / n

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or no_improve == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE {train_loss / n_total:.5f} | "
                f"val MSE {val_loss:.5f} | "
                f"lr {sched.get_last_lr()[0]:.2e}"
            )

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val MSE: {best_val:.5f}"
            )
            break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Calculate validation R² with best model
    model.eval()
    val_preds_final = []
    val_targets_final = []
    val_mse_final = 0.0
    n_val = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_mse_final += nn.functional.mse_loss(
                pred, yb, reduction="sum"
            ).item()
            n_val += xb.size(0)
            val_preds_final.append(pred.cpu())
            val_targets_final.append(yb.cpu())

    val_mse_final = val_mse_final / n_val
    val_preds_concat = torch.cat(val_preds_final, dim=0).squeeze().numpy()
    val_targets_concat = torch.cat(val_targets_final, dim=0).squeeze().numpy()

    # Denormalize for R² calculation (in original units)
    val_preds_orig = invert_y_norm(val_preds_concat, stats)
    val_targets_orig = invert_y_norm(val_targets_concat, stats)

    # Calculate R² manually: 1 - (SS_res / SS_tot)
    ss_res = np.sum((val_targets_orig - val_preds_orig) ** 2)
    ss_tot = np.sum((val_targets_orig - np.mean(val_targets_orig)) ** 2)
    val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Evaluate on test set (still in normalised units)
    model.eval()
    test_preds = []
    test_targets = []
    mse = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse += nn.functional.mse_loss(
                pred, yb, reduction="sum"
            ).item()
            n += xb.size(0)
            test_preds.append(pred.cpu())
            test_targets.append(yb.cpu())

    test_mse = mse / n
    test_rmse = math.sqrt(test_mse)

    # Calculate test R² (in original units)
    test_preds_concat = torch.cat(test_preds, dim=0).squeeze().numpy()
    test_targets_concat = torch.cat(test_targets, dim=0).squeeze().numpy()
    test_preds_orig = invert_y_norm(test_preds_concat, stats)
    test_targets_orig = invert_y_norm(test_targets_concat, stats)

    ss_res_test = np.sum((test_targets_orig - test_preds_orig) ** 2)
    ss_tot_test = np.sum((test_targets_orig - np.mean(test_targets_orig)) ** 2)
    test_r2 = 1 - (ss_res_test / ss_tot_test) if ss_tot_test > 0 else 0.0

    print(f"Test RMSE (normalized target units): {test_rmse:.5f}")
    print(f"Validation R²: {val_r2:.5f}")
    print(f"Test R²: {test_r2:.5f}")

    return model.cpu(), stats, (val_mse_final, val_r2, test_mse, test_r2)


# ----------------------------------------------------------------------
# Save / Load
# ----------------------------------------------------------------------


def save_model(model: MLP, stats: NormStats, path: str = "interpolator.pkl") -> None:
    payload = {
        "state_dict": model.state_dict(),
        "norm_stats": asdict(stats),
        "meta": {"in_dim": 5, "hidden": 64, "depth": 3, "out_dim": 1},
        "model_info": {
            "framework": "pytorch",
            "version": torch.__version__,
            "saved_at": str(np.datetime64("now")),
        },
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved model to {path}")


def load_model(path: str = "interpolator.pkl") -> Tuple[MLP, NormStats]:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    meta = payload["meta"]
    model = MLP(
        in_dim=meta["in_dim"],
        hidden=meta["hidden"],
        depth=meta["depth"],
        out_dim=meta["out_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    stats = NormStats(**payload["norm_stats"])
    model.eval()
    return model, stats


# ----------------------------------------------------------------------
# Interpolate new points
# X_new: (N, 5) numpy in original units
# returns y_pred in original units
# ----------------------------------------------------------------------


def interpolate(
    model: MLP,
    stats: NormStats,
    X_new: np.ndarray,
) -> np.ndarray:
    Xn = apply_x_norm(X_new.astype(np.float32), stats)
    with torch.no_grad():
        yn = model(torch.from_numpy(Xn).float()).squeeze(1).numpy()
    return invert_y_norm(yn, stats)


# ----------------------------------------------------------------------
# Synthetic data generation utilities
# ----------------------------------------------------------------------


def synthetic_5d(n: int = 5000, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple synthetic 5D regression problem.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 5)).astype(np.float32)

    x0, x1, x2, x3, x4 = X.T
    y = (
        np.sin(2 * np.pi * x0)
        + 0.5 * x1**2
        - 0.3 * x2
        + np.cos(x3 * x4 * np.pi)
    ).astype(np.float32)

    return X, y


def generate_synthetic_pkl(
    n: int = 5000,
    seed: int = 123,
    workdir: str = ".",
    filename: str = "synthetic_5d_data.pkl",
) -> str:
    """
    Generate synthetic 5D data and save to pickle file.

    """
    os.makedirs(workdir, exist_ok=True)
    filepath = os.path.join(workdir, filename)

    print(f"Generating {n} synthetic 5D data points...")
    X, y = synthetic_5d(n=n, seed=seed)

    with open(filepath, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    print(f"Saved synthetic data to {filepath}")
    return filepath
