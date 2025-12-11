"""
5D dataset interpolator with neural network regression.
This module provides both:
1. Sklearn-style interface (NeuralNetworkRegressor) for Jupyter notebooks
2. Function-based interface (train_model, interpolate) for web API
"""

import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

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


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination) score.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        R² score (1.0 is perfect prediction)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


# ----------------------------------------------------------------------
# Configurable MLP with variable layer sizes
# ----------------------------------------------------------------------


class ConfigurableMLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable layer sizes.

    Supports variable neurons per layer (e.g., [64, 32, 16]).
    This satisfies coursework requirement for configurable architecture.
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden_layers: List[int] = None,
        out_dim: int = 1,
        activation: str = "silu",
        hidden: int = None,  # Backward compatibility
        depth: int = None,   # Backward compatibility
        act = None,          # Backward compatibility
    ):
        """
        Initialize MLP with configurable layer sizes.

        Args:
            in_dim: Input dimension (number of features)
            hidden_layers: List of hidden layer sizes (e.g., [64, 32, 16])
            out_dim: Output dimension
            activation: Activation function ('silu', 'relu', 'tanh')
            hidden: (deprecated) Uniform hidden size for all layers
            depth: (deprecated) Number of hidden layers
            act: (deprecated) Activation class
        """
        super().__init__()

        # Handle backward compatibility: if old params provided, convert to new format
        if hidden is not None and depth is not None:
            hidden_layers = [hidden] * depth
        elif hidden_layers is None:
            hidden_layers = [64, 32, 16]

        # Handle activation backward compatibility
        if act is not None:
            # Use old-style activation class directly
            act_class = act
        else:
            # Use new-style string activation
            act_map = {
                "silu": nn.SiLU,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
            }
            act_class = act_map.get(activation, nn.SiLU)

        # Build network with varying layer sizes
        layers = []
        prev_size = in_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_class())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, out_dim))

        self.net = nn.Sequential(*layers)
        self.hidden_layers = hidden_layers  # Store for saving/loading
        self.activation = activation  # Store activation string for saving/loading

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# Backward compatibility alias
MLP = ConfigurableMLP


# ----------------------------------------------------------------------
# Scikit-learn compatible wrapper for Jupyter notebook usage
# ----------------------------------------------------------------------


class NeuralNetworkRegressor:
    """
    Scikit-learn compatible neural network regressor.

    Provides standard fit(X, y) and predict(X) interface as required
    by the coursework specification.

    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        learning_rate: float = 5e-3,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        weight_decay: float = 1e-6,
        activation: str = "silu",
        random_state: int = 123,
        verbose: bool = True,
    ):
        """
        Initialize the neural network regressor.
        """
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.activation = activation
        self.random_state = random_state
        self.verbose = verbose

        # Will be set during fit
        self.model_: Optional[ConfigurableMLP] = None
        self.norm_stats_: Optional[NormStats] = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training history
        self.history_ = {
            "train_loss": [],
            "val_loss": [],
            "epochs_trained": 0,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkRegressor":
        """
        Fit the neural network to training data.
        """
        # Use train_model() to avoid code duplication
        # train_model() handles all the training logic and returns metrics
        self.model_, self.norm_stats_, (val_mse, val_r2, test_mse, test_r2) = train_model(
            X=X,
            y=y,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Move model to appropriate device
        self.model_ = self.model_.to(self.device_)

        # Store metrics in history for user access
        self.history_["val_mse"] = val_mse
        self.history_["val_r2"] = val_r2
        self.history_["test_mse"] = test_mse
        self.history_["test_r2"] = test_r2

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        """
        if self.model_ is None or self.norm_stats_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Validate and normalize input
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 5:
            raise ValueError(f"Expected 5 features, got {X.shape[1]}")

        Xn = apply_x_norm(X, self.norm_stats_)  # Already float32 from X

        # Predict
        self.model_.eval()
        with torch.no_grad():
            Xn_t = torch.from_numpy(Xn).to(self.device_)
            yn_pred = self.model_(Xn_t).cpu().squeeze().numpy()

        # Denormalize
        y_pred = invert_y_norm(yn_pred, self.norm_stats_)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.
        """
        y_pred = self.predict(X)
        return calculate_r2(y, y_pred)


# ----------------------------------------------------------------------
# Training loop with early stopping (for web API)
# ----------------------------------------------------------------------


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    max_epochs: int = 200,
    lr: float = 5e-3,
    weight_decay: float = 1e-6,
    patience: int = 20,
    hidden_layers: List[int] = None,
    activation: str = "silu",
    random_state: int = 123,
    verbose: bool = True,
) -> Tuple[ConfigurableMLP, NormStats, Tuple[float, float, float, float]]:
    """
    Train a neural network model on the provided data.
    """
    if verbose:
        print("=" * 60)
        print("🚀 TRAINING MODEL")
        print("=" * 60)

    if hidden_layers is None:
        hidden_layers = [64, 32, 16]

    # Compute normalization and normalize data
    stats = compute_norm_stats(X, y)
    Xn = apply_x_norm(X, stats)  # Already float32 from compute_norm_stats
    yn = apply_y_norm(y, stats)  # Already float32 from compute_norm_stats

    Xn_t = torch.from_numpy(Xn)
    yn_t = torch.from_numpy(yn).unsqueeze(1)

    ds = TensorDataset(Xn_t, yn_t)

    n_total = len(ds)
    # Use 70% train, 15% validation, 15% test split
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    n_train = n_total - n_val - n_test

    # Ensure we have at least some samples in each split
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        # For very small datasets, ensure at least 1 sample per split
        n_val = max(1, int(0.15 * n_total))
        n_test = max(1, int(0.15 * n_total))
        n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConfigurableMLP(
        in_dim=5,
        hidden_layers=hidden_layers,
        out_dim=1,
        activation=activation,
    ).to(device)

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
        val_loss = 0.0
        n_val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += nn.functional.mse_loss(
                    pred, yb, reduction="sum"
                ).item()
                n_val_count += xb.size(0)
        val_loss = val_loss / n_val_count

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or no_improve == 1):
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE {train_loss / n_total:.5f} | "
                f"val MSE {val_loss:.5f} | "
                f"lr {sched.get_last_lr()[0]:.2e}"
            )

        if no_improve >= patience:
            if verbose:
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
    n_val_samples = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_mse_final += nn.functional.mse_loss(
                pred, yb, reduction="sum"
            ).item()
            n_val_samples += xb.size(0)
            val_preds_final.append(pred.cpu())
            val_targets_final.append(yb.cpu())

    val_mse_final = val_mse_final / n_val_samples
    val_preds_concat = torch.cat(val_preds_final, dim=0).squeeze().numpy()
    val_targets_concat = torch.cat(val_targets_final, dim=0).squeeze().numpy()

    # Denormalize for R² calculation (in original units)
    val_preds_orig = invert_y_norm(val_preds_concat, stats)
    val_targets_orig = invert_y_norm(val_targets_concat, stats)

    # Calculate R² score
    val_r2 = calculate_r2(val_targets_orig, val_preds_orig)

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

    test_r2 = calculate_r2(test_targets_orig, test_preds_orig)

    if verbose:
        print(f"Test RMSE (normalized target units): {test_rmse:.5f}")
        print(f"Validation R²: {val_r2:.5f}")
        print(f"Test R²: {test_r2:.5f}")

    return model.cpu(), stats, (val_mse_final, val_r2, test_mse, test_r2)


# ----------------------------------------------------------------------
# Save / Load
# ----------------------------------------------------------------------


def save_model(model: ConfigurableMLP, stats: NormStats, path: str = "interpolator.pkl") -> None:
    """
    Save trained model with normalization statistics.
    """
    # Extract architecture from model
    hidden_layers = getattr(model, 'hidden_layers', [64, 32, 16])
    activation = getattr(model, 'activation', 'silu')

    payload = {
        "state_dict": model.state_dict(),
        "norm_stats": asdict(stats),
        "meta": {
            "in_dim": 5,
            "hidden_layers": hidden_layers,
            "out_dim": 1,
            "activation": activation,
        },
        "model_info": {
            "framework": "pytorch",
            "version": torch.__version__,
            "saved_at": str(np.datetime64("now")),
        },
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved model to {path}")


def load_model(path: str = "interpolator.pkl") -> Tuple[ConfigurableMLP, NormStats]:
    """
    Load trained model with normalization statistics.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    meta = payload["meta"]

    # Handle both old and new formats
    if "hidden_layers" in meta:
        # New format with variable layer sizes
        model = ConfigurableMLP(
            in_dim=meta["in_dim"],
            hidden_layers=meta["hidden_layers"],
            out_dim=meta["out_dim"],
            activation=meta.get("activation", "silu"),
        )
    else:
        # Old format - convert to new format
        hidden = meta.get("hidden", 64)
        depth = meta.get("depth", 3)
        hidden_layers = [hidden] * depth
        model = ConfigurableMLP(
            in_dim=meta["in_dim"],
            hidden_layers=hidden_layers,
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
    model: ConfigurableMLP,
    stats: NormStats,
    X_new: np.ndarray,
) -> np.ndarray:
    """
    Make predictions on new data points.

    Args:
        model: Trained model
        stats: Normalization statistics
        X_new: New input features (N, 5)

    Returns:
        Predictions in original units
    """
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
