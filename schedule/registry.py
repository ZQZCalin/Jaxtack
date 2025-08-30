# Learning rate schedule registry.


from common import _type
from schedule.config import (
    TrapezoidScheduleConfig
)
from schedule.base import trapezoid_schedule


def schedule_registry() -> _type.LearningRate:
    