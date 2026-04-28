# base.py
"""Base classes and types."""

import jax
import optax


Array = jax.Array
PRNGKeyArray = jax.Array
ArrayLike = jax.typing.ArrayLike

Optimizer = optax.GradientTransformation
Schedule = optax.ScalarOrSchedule
