from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fivedreg.interpolator import MLP, NormStats

@dataclass
class GlobalState:
    model: "MLP | None" = None
    norm_stats: "NormStats | None" = None
    last_metrics: dict | None = None

STATE = GlobalState()

