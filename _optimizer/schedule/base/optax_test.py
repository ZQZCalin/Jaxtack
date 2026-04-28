"""Tests for LEARNING_RATE_REGISTRY and schedule builders in optax.py"""

import pytest
from _optimizer.schedule.registry import LEARNING_RATE_REGISTRY
from _optimizer.schedule.base import optax as schedule_optax


# Schedule names registered in optax.py (name -> config class for default instantiation)
SCHEDULE_CONFIGS = [
    ("constant_schedule", schedule_optax.ConstantScheduleConfig),
    ("warmup_constant_schedule", schedule_optax.WarmupConstantScheduleConfig),
    ("cosine_decay_schedule", schedule_optax.CosineDecayScheduleConfig),
    ("cosine_onecycle_schedule", schedule_optax.CosineOnecycleScheduleConfig),
    ("warmup_cosine_decay_schedule", schedule_optax.WarmupCosineDecayScheduleConfig),
    ("exponential_decay", schedule_optax.ExponentialDecayScheduleConfig),
    ("warmup_exponential_decay_schedule", schedule_optax.WarmupExponentialDecayScheduleConfig),
    ("linear_onecycle_schedule", schedule_optax.LinearOnecycleScheduleConfig),
    ("linear_schedule", schedule_optax.LinearScheduleConfig),
    ("piecewise_constant_schedule", schedule_optax.PiecewiseConstantScheduleConfig),
    ("piecewise_interpolate_schedule", schedule_optax.PiecewiseInterpolateScheduleConfig),
    ("polynomial_schedule", schedule_optax.PolynomialScheduleConfig),
    ("sgdr_schedule", schedule_optax.SGDRScheduleConfig),
]


class TestLearningRateRegistry:
    def test_list_available_contains_all_expected(self):
        available = LEARNING_RATE_REGISTRY.list_available()
        expected_names = [name for name, _ in SCHEDULE_CONFIGS]
        for name in expected_names:
            assert name in available, f"Expected '{name}' in registry"

    @pytest.mark.parametrize("name,config_cls", SCHEDULE_CONFIGS)
    def test_get_config_class_returns_valid_class(self, name, config_cls):
        retrieved = LEARNING_RATE_REGISTRY.get_config_class(name)
        assert retrieved is config_cls
        config = config_cls(name=name)
        assert config.name == name

    @pytest.mark.parametrize("name,config_cls", SCHEDULE_CONFIGS)
    def test_build_and_run_schedule_with_default_config(self, name, config_cls):
        """Build each schedule with default config and verify it runs (callable, no error)."""
        config = config_cls(name=name)
        schedule = LEARNING_RATE_REGISTRY.build(config)
        assert callable(schedule)
        # Toy test
        _ = schedule(0)
        _ = schedule(100)
