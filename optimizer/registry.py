# Optimizer registry.


from optimizer.config import (
    OptaxAdamWConfig
)
from optimizer.optax import (

)
from common import _type


def optimizer_registry(
        
) -> _type.Optimizer:
    name = config.name
    optim_config = config.optim_config
    schedule = schedule_registry(

    )
    if name == "optax_adamw":
        optim = OptaxAdamW