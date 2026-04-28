# trainer/config.py

from pydantic import BaseModel
from typing import NamedTuple
from jax import Array
from jax.typing import ArrayLike


class BaseTrainerConfig(BaseModel):
    name: str


class BaseTrainerState(NamedTuple):
    """Base train state class is a serializable class to wrap all training states, including:
    - model state
    - optimizer state
    - root random key in training
    - any other states that are needed for training 
        (such as epoch and iteration counter,logger, checkpoint manager, etc.)
    """
    model: ArrayLike
    opt_state: ArrayLike
    prng_key: Array | None = None