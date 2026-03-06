# Wrapper for optax-implemented optimizers.

from typing import Any, Callable
from jax.typing import ArrayLike
from optimizer.config import BaseOptimizerConfig

import optax
from optimizer.registry import OPTIMIZER_REGISTRY


class AdamWConfig(BaseOptimizerConfig):
    """Optax-implemented AdamW.
    
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw
    """
    name: str = "adamw"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-08
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    weight_decay: ArrayLike = 0.0001
    mask: Any | Callable[[optax.Params], Any] | None = None
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("adamw", AdamWConfig)
def build_adamw(
        config: AdamWConfig,
        learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        weight_decay=config.weight_decay,
        mask=config.mask,
        nesterov=config.nesterov,
    )