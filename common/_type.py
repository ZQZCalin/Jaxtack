# Common types

import jax
import optax
from typing import Any, Callable


Model = jax.Module
Dataset = Any
Optimizer = optax.GradientTransform
Schedule = optax.ScalarOrSchedule
Config = Any

ModelCls = Callable[..., Model]
DatasetCls = Callable[..., Dataset]
OptimizerCls = Callable[..., Optimizer]
ScheduleCls = Callable[..., Schedule]
ConfigCls = Callable[..., Config]
