# Config dataclasses for learing rate schedules.


from dataclasses import dataclass


@dataclass
class TrapezoidScheduleConfig:
    peak_value: float
    total_steps: int
    warmup_steps: int = 0
    decay_steps: int = 0
