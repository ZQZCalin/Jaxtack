from omegaconf import OmegaConf, DictConfig

import optax
from .registry import OPTIMIZER_REGISTRY

def build_optimizer(optimizer_config: DictConfig) -> optax.GradientTransformation:
    """Build an optimizer from .yaml config."""
    name: str = optimizer_config.name
    
    ConfigClass = OPTIMIZER_REGISTRY.get_config_class(name)
    config = ConfigClass(**OmegaConf.to_container(optimizer_config, resolve=True))

    # NOTE: user may customize building process for special optimizers.
    # if name == "my_optim":
    #     mask = ...
    #     optimizer = OPTIMIZER_REGISTRY.build(
    #         config, 
    #         mask=mask
    #     )
    #     return optimizer

    # By default, we unpack the learning rate schedule.
    learning_rate_config = optimizer_config.get("learning_rate", None)
    learning_rate = XXX # TODO
    optimizer = OPTIMIZER_REGISTRY.build(
        config, 
        learning_rate=learning_rate
    )
    return optimizer