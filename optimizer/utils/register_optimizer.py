# Main register optimizer function.


import optax

from common import _type


def register_optimizer(
        name: str,
        optim: _type.Optim,
        config: _type.Config,
):
    """Register an optimizer for initialization."""
    