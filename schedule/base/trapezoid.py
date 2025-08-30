# The trapezoid schedule.

import optax

def trapezoid_schedule(
        peak_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 0,
) -> optax.Schedule:
    """Implements the trapezoid schedule
    
    https://arxiv.org/pdf/2405.18392

    Args:
        peak_value: maximum lr.
        total_steps: total number of steps, including warmup and decay.
        warmup_steps: number of warmup steps.
        decay_steps: number of decay steps.

    Returns:
        An `optax.Schedule` object
    """

    schedules = [
        optax.linear_schedule(
            init_value=0.0,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=peak_value,
            transition_steps=total_steps - warmup_steps - decay_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=0.0,
            transition_steps=decay_steps,
        )
    ]
    return optax.join_schedules(schedules, [warmup_steps, total_steps - decay_steps])