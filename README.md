# JaxTack

## Customization

In JaxTack, all training components are customizable, including model, dataset, optimizer, training loop, etc., so long as certain simple rules are followed.

Rule 1: All config classes only specify hyper-parameters, so the types are `str, int, float, bool`. Any other more complicated 

Rule 2: 

```py
from .config import BaseOptimizerConfig

class MyOptim
```

### Optimizer

1. User must provide an optimizer building function of form 
    ```py
    def init_optim(config: BaseOptimizerConfig, learning_rate: optax.ScalarOrSchedule | None, **kwargs) -> optax.GradientTransformation:
        return optax.adam(
            learning_rate=learning_rate,
            b1=config.b1,
            ...
        )
    ```
2. It is recommended to provide a matched pair of
    i. a pydantic config class inheriting `BaseOptimizerConfig`
    ii. an optimizer function
    For example,
    ```py
    class MyOptimConfig(BaseOptimizerConfig):
        name: str = "my_optim"
        arg1: float = 0.0
        ...

    def my_optim(arg1: float, ...) -> optax.GradientTransformation:
        ...
    ```
3. By default, `init_optimzier` will unpack the learning rate schedule for you. Therefore, if learning rate is the only non-hyper-parameter for the optimizer, you are good to go! However, in case you need other arguments to construct your optimizer, you can go to `init_optimizer.py` and add additional section to unpack extra args
    ```py
    if config.name == "my_optim":
        mask = ...
        optimizer = OPTIMIZER_REGISTRY.build(
            opt_cfg, schedule=schedule, mask=mask
        )
    ```