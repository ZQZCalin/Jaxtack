# Learning rate schedule registry.


from common import _type
from common.registry import Registry
from schedule.config import (
    TrapezoidScheduleConfig
)
from schedule.base import trapezoid_schedule


schedule_registry = Registry(module_name="Schedule")


def register_schedule(name: str, schedule_class: _type.ScheduleClass,
                      config_class: _type.ConfigClass) -> None:
    schedule_registry.register(name, schedule_class, config_class)


def build_schedule(name: str, config: _type.Config):
    return schedule_registry.build(name, config)


# ==============================================================================
# Schedule Registry: add your schedule below...
# ==============================================================================

register_schedule(
    name="trapezoid", 
    schedule_class=trapezoid_schedule, 
    config_class=TrapezoidScheduleConfig,
)

