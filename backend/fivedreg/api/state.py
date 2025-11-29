from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler
    from fivedreg.models.mlp import MLPRegressor

@dataclass
class GlobalState:
    model: "MLPRegressor | None" = None
    x_scaler: "StandardScaler | None" = None
    y_scaler: "StandardScaler | None" = None
    last_metrics: dict | None = None

STATE = GlobalState()

