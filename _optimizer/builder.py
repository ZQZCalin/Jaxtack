# Optimizer builder.

import optax
from omegaconf import OmegaConf, DictConfig
from _optimizer.registry import OPTIMIZER_REGISTRY
from _optimizer.schedule.registry import build_learning_rate


# TODO: encourage customized config unpacking for special optimizers,
# and include them in the same optimizer file.

def build_optimizer(cfg: DictConfig) -> optax.GradientTransformation:
    """Wrapper for building optimizer from yaml config."""
    name: str = cfg.name
    
    ConfigClass = OPTIMIZER_REGISTRY.get_config_class(name)
    optimizer_config = ConfigClass(**OmegaConf.to_container(cfg, resolve=True))

    # NOTE: user may customize building process for special optimizers.
    # if name == "my_optim":
    #     mask = ...
    #     optimizer = OPTIMIZER_REGISTRY.build(
    #         optimizer_config, 
    #         mask=mask
    #     )
    #     return optimizer

    # By default, we unpack the learning rate schedule.
    learning_rate_config = cfg.get("learning_rate", None)
    learning_rate = build_learning_rate(learning_rate_config)
    optimizer = OPTIMIZER_REGISTRY.build(
        optimizer_config, 
        learning_rate=learning_rate
    )
    return optimizer