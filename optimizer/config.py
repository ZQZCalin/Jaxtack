# Optimizer config classes.

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import optax


@dataclass
class OptaxAdamWConfig:
    """Optax-implemented AdamW.
    
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw
    """
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-08
    eps_root: float = 0.0
    mu_dtype: Any | None = None
    weight_decay: float = 0.0001
    mask: Any | Callable[[optax.Params], Any] | None = None
    nesterov: bool = False


