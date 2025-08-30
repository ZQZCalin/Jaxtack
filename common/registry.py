# Register functions for models, datasets, and optimizers.

from typing import Dict, Tuple
from common import _type

class ScheduleRegistry:
    def __init__(self) -> None:
        self._map: Dict[str, Tuple[_type.ScheduleCls, _type.ConfigCls]] = {}

    def register(self, name: str, schedule_cls: _type.ScheduleCls, config_cls: _type.ConfigCls) -> None:
        if name in self._map:
            raise KeyError(f"Schedule '{name}' already registered.")
        self._map[name] = (schedule_cls, config_cls)

    def build(self, name: str, config: _type.Config) -> _type.Schedule:
        pass


schedule_registry = ScheduleRegistry()

def register_schedule(
        name: str,
        schedule_cls: _type.Schedule,
        config_cls: _type.Config,
) -> None:
    """Register a schedule class.
    
    Args:
        - name: name of learning rate schedule.
        - schedule: class that defines the schedule.
        - config: config dataclass.
    """
    schedule_registry.register(name, schedule, config)

def build_schedule(
        name: str,
        config: _type.Config,
)