# Common types

import jax
import optax
from typing import Any, Callable


Model = jax.Module
Dataset = Any
Optimizer = optax.GradientTransform
Schedule = optax.ScalarOrSchedule
Config = Any

ModelClass = Callable[..., Model]
DatasetClass = Callable[..., Dataset]
OptimizerClass = Callable[..., Optimizer]
ScheduleClass = Callable[..., Schedule]
ConfigClass = Callable[..., Config]
