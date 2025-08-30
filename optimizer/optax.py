# Wrapper for optax-implemented optimizers.

import optax
from common import _type
from optimizer.config import OptaxAdamWConfig


def optax_adamw(
        learning_rate: _type.LearningRate,
        config: OptaxAdamWConfig,
) -> _type.Optim:
    """Wrapper for optax.adamw"""
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