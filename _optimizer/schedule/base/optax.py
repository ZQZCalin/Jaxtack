# Wrapper for optax learning rate schedules.
# https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html

from __future__ import annotations

from typing import Any
from _optimizer.schedule.registry import (
    LEARNING_RATE_REGISTRY,
    BaseLearningRateConfig,
)

import optax


# -----------------------------------------------------------------------------
# Constant schedule
# -----------------------------------------------------------------------------
class ConstantScheduleConfig(BaseLearningRateConfig):
    """Optax constant schedule.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.constant_schedule
    """
    name: str = "constant_schedule"
    value: float = 1e-3


@LEARNING_RATE_REGISTRY.register("constant_schedule", ConstantScheduleConfig)
def build_constant_schedule(config: ConstantScheduleConfig) -> optax.ScalarOrSchedule:
    return optax.constant_schedule(config.value)


# -----------------------------------------------------------------------------
# Warmup constant schedule
# -----------------------------------------------------------------------------
class WarmupConstantScheduleConfig(BaseLearningRateConfig):
    """Optax linear warmup then constant.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_constant_schedule
    """
    name: str = "warmup_constant_schedule"
    init_value: float = 0.0
    peak_value: float = 1e-3
    warmup_steps: int = 1000


@LEARNING_RATE_REGISTRY.register("warmup_constant_schedule", WarmupConstantScheduleConfig)
def build_warmup_constant_schedule(
    config: WarmupConstantScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.warmup_constant_schedule(
        init_value=config.init_value,
        peak_value=config.peak_value,
        warmup_steps=config.warmup_steps,
    )


# -----------------------------------------------------------------------------
# Cosine decay schedule
# -----------------------------------------------------------------------------
class CosineDecayScheduleConfig(BaseLearningRateConfig):
    """Optax cosine decay.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.cosine_decay_schedule
    """
    name: str = "cosine_decay_schedule"
    init_value: float = 1e-3
    decay_steps: int = 10000
    alpha: float = 0.0
    exponent: float = 1.0


@LEARNING_RATE_REGISTRY.register("cosine_decay_schedule", CosineDecayScheduleConfig)
def build_cosine_decay_schedule(
    config: CosineDecayScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.cosine_decay_schedule(
        init_value=config.init_value,
        decay_steps=config.decay_steps,
        alpha=config.alpha,
        exponent=config.exponent,
    )


# -----------------------------------------------------------------------------
# Cosine onecycle schedule
# -----------------------------------------------------------------------------
class CosineOnecycleScheduleConfig(BaseLearningRateConfig):
    """Optax cosine onecycle.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.cosine_onecycle_schedule
    """
    name: str = "cosine_onecycle_schedule"
    transition_steps: int = 10000
    peak_value: float = 1e-3
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 10000.0


@LEARNING_RATE_REGISTRY.register("cosine_onecycle_schedule", CosineOnecycleScheduleConfig)
def build_cosine_onecycle_schedule(
    config: CosineOnecycleScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.cosine_onecycle_schedule(
        transition_steps=config.transition_steps,
        peak_value=config.peak_value,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )


# -----------------------------------------------------------------------------
# Warmup cosine decay schedule
# -----------------------------------------------------------------------------
class WarmupCosineDecayScheduleConfig(BaseLearningRateConfig):
    """Optax linear warmup + cosine decay.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_cosine_decay_schedule
    """
    name: str = "warmup_cosine_decay_schedule"
    init_value: float = 0.0
    peak_value: float = 1e-3
    warmup_steps: int = 1000
    decay_steps: int = 10000
    end_value: float = 0.0
    exponent: float = 1.0


@LEARNING_RATE_REGISTRY.register(
    "warmup_cosine_decay_schedule", WarmupCosineDecayScheduleConfig
)
def build_warmup_cosine_decay_schedule(
    config: WarmupCosineDecayScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=config.init_value,
        peak_value=config.peak_value,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.end_value,
        exponent=config.exponent,
    )


# -----------------------------------------------------------------------------
# Exponential decay schedule
# -----------------------------------------------------------------------------
class ExponentialDecayScheduleConfig(BaseLearningRateConfig):
    """Optax exponential decay.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.exponential_decay
    """
    name: str = "exponential_decay"
    init_value: float = 1e-3
    transition_steps: int = 1000
    decay_rate: float = 0.9
    transition_begin: int = 0
    staircase: bool = False
    end_value: float | None = None


@LEARNING_RATE_REGISTRY.register("exponential_decay", ExponentialDecayScheduleConfig)
def build_exponential_decay_schedule(
    config: ExponentialDecayScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.exponential_decay(
        init_value=config.init_value,
        transition_steps=config.transition_steps,
        decay_rate=config.decay_rate,
        transition_begin=config.transition_begin,
        staircase=config.staircase,
        end_value=config.end_value,
    )


# -----------------------------------------------------------------------------
# Warmup exponential decay schedule
# -----------------------------------------------------------------------------
class WarmupExponentialDecayScheduleConfig(BaseLearningRateConfig):
    """Optax linear warmup + exponential decay.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_exponential_decay_schedule
    """
    name: str = "warmup_exponential_decay_schedule"
    init_value: float = 0.0
    peak_value: float = 1e-3
    warmup_steps: int = 1000
    transition_steps: int = 10000
    decay_rate: float = 0.9
    transition_begin: int = 0
    staircase: bool = False
    end_value: float | None = None


@LEARNING_RATE_REGISTRY.register(
    "warmup_exponential_decay_schedule", WarmupExponentialDecayScheduleConfig
)
def build_warmup_exponential_decay_schedule(
    config: WarmupExponentialDecayScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.warmup_exponential_decay_schedule(
        init_value=config.init_value,
        peak_value=config.peak_value,
        warmup_steps=config.warmup_steps,
        transition_steps=config.transition_steps,
        decay_rate=config.decay_rate,
        transition_begin=config.transition_begin,
        staircase=config.staircase,
        end_value=config.end_value,
    )


# join_schedules(schedules, boundaries) takes callables; not config-driven here.
# Use optax.join_schedules(schedules, boundaries) programmatically if needed.

# -----------------------------------------------------------------------------
# Linear onecycle schedule
# -----------------------------------------------------------------------------
class LinearOnecycleScheduleConfig(BaseLearningRateConfig):
    """Optax linear onecycle (three linear phases).
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.linear_onecycle_schedule
    """
    name: str = "linear_onecycle_schedule"
    transition_steps: int = 10000
    peak_value: float = 1e-3
    pct_start: float = 0.3
    pct_final: float = 0.85
    div_factor: float = 25.0
    final_div_factor: float = 10000.0


@LEARNING_RATE_REGISTRY.register(
    "linear_onecycle_schedule", LinearOnecycleScheduleConfig
)
def build_linear_onecycle_schedule(
    config: LinearOnecycleScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.linear_onecycle_schedule(
        transition_steps=config.transition_steps,
        peak_value=config.peak_value,
        pct_start=config.pct_start,
        pct_final=config.pct_final,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )


# -----------------------------------------------------------------------------
# Linear schedule
# -----------------------------------------------------------------------------
class LinearScheduleConfig(BaseLearningRateConfig):
    """Optax linear schedule.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.linear_schedule
    """
    name: str = "linear_schedule"
    init_value: float = 1e-3
    end_value: float = 1e-5
    transition_steps: int = 10000
    transition_begin: int = 0


@LEARNING_RATE_REGISTRY.register("linear_schedule", LinearScheduleConfig)
def build_linear_schedule(
    config: LinearScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.linear_schedule(
        init_value=config.init_value,
        end_value=config.end_value,
        transition_steps=config.transition_steps,
        transition_begin=config.transition_begin,
    )


# -----------------------------------------------------------------------------
# Piecewise constant schedule
# -----------------------------------------------------------------------------
class PiecewiseConstantScheduleConfig(BaseLearningRateConfig):
    """Optax piecewise constant schedule.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.piecewise_constant_schedule
    """
    name: str = "piecewise_constant_schedule"
    init_value: float = 1e-3
    boundaries_and_scales: dict[int, float] | dict[str, float] | None = None


@LEARNING_RATE_REGISTRY.register(
    "piecewise_constant_schedule", PiecewiseConstantScheduleConfig
)
def build_piecewise_constant_schedule(
    config: PiecewiseConstantScheduleConfig,
) -> optax.ScalarOrSchedule:
    bas = config.boundaries_and_scales
    if bas is not None:
        bas = {int(k): float(v) for k, v in bas.items()}
    return optax.piecewise_constant_schedule(
        init_value=config.init_value,
        boundaries_and_scales=bas,
    )


# -----------------------------------------------------------------------------
# Piecewise interpolate schedule
# -----------------------------------------------------------------------------
class PiecewiseInterpolateScheduleConfig(BaseLearningRateConfig):
    """Optax piecewise interpolate (linear or cosine between boundaries).
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.piecewise_interpolate_schedule
    """
    name: str = "piecewise_interpolate_schedule"
    interpolate_type: str = "linear"  # "linear" or "cosine"
    init_value: float = 1e-3
    boundaries_and_scales: dict[int, float] | dict[str, float] | None = None


@LEARNING_RATE_REGISTRY.register(
    "piecewise_interpolate_schedule", PiecewiseInterpolateScheduleConfig
)
def build_piecewise_interpolate_schedule(
    config: PiecewiseInterpolateScheduleConfig,
) -> optax.ScalarOrSchedule:
    bas = config.boundaries_and_scales
    if bas is not None:
        bas = {int(k): float(v) for k, v in bas.items()}
    return optax.piecewise_interpolate_schedule(
        interpolate_type=config.interpolate_type,
        init_value=config.init_value,
        boundaries_and_scales=bas,
    )


# -----------------------------------------------------------------------------
# Polynomial schedule
# -----------------------------------------------------------------------------
class PolynomialScheduleConfig(BaseLearningRateConfig):
    """Optax polynomial schedule.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.polynomial_schedule
    """
    name: str = "polynomial_schedule"
    init_value: float = 1e-3
    end_value: float = 1e-5
    power: float = 2.0
    transition_steps: int = 10000
    transition_begin: int = 0


@LEARNING_RATE_REGISTRY.register("polynomial_schedule", PolynomialScheduleConfig)
def build_polynomial_schedule(
    config: PolynomialScheduleConfig,
) -> optax.ScalarOrSchedule:
    return optax.polynomial_schedule(
        init_value=config.init_value,
        end_value=config.end_value,
        power=config.power,
        transition_steps=config.transition_steps,
        transition_begin=config.transition_begin,
    )


# -----------------------------------------------------------------------------
# SGDR schedule (warm restarts)
# -----------------------------------------------------------------------------
class SGDRScheduleConfig(BaseLearningRateConfig):
    """Optax SGDR (cosine decay with warm restarts).
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.sgdr_schedule

    cosine_kwargs: list of dicts, each with init_value, peak_value, warmup_steps,
    decay_steps, and optionally end_value, exponent (for each cycle).
    """
    name: str = "sgdr_schedule"
    cosine_kwargs: list[dict[str, Any]] = []


@LEARNING_RATE_REGISTRY.register("sgdr_schedule", SGDRScheduleConfig)
def build_sgdr_schedule(config: SGDRScheduleConfig) -> optax.ScalarOrSchedule:
    return optax.sgdr_schedule(cosine_kwargs=config.cosine_kwargs)
