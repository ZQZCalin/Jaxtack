# Learning rate schedule registry.


from common import _type
from schedule.config import (
    TrapezoidScheduleConfig
)
from schedule.base import trapezoid_schedule


register_schedule(name="trapezoid", schedule=trapezoid_schedule, config=TrapezoidScheduleConfig)