# Register functions for models, datasets, and optimizers.

from typing import Dict, Tuple, Callable, Any
from common import _type
import inspect
from dataclasses import fields, is_dataclass, asdict

import logging
logger: logging.Logger = logging.Logger(name=__name__)
logger.setLevel(logging.WARNING)


def _check_arg_match(
        func_class: Callable[..., Any], 
        config_class: _type.ConfigClass,
) -> None:
    """Checks whether argument of a function matches a dataclass.
    Positional args and keyword args are neglected.
    
    Args:
        - func_class: a callable function
        - config_class: a dataclass object that contains function arguments

    Raises:
        ValueError if arguments mismatch
    """
    # Function arguments.
    func_args = set()
    sig = inspect.signature(func_class)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL, 
            inspect.Parameter.VAR_KEYWORD
        ):
            logger.warning(
                f"'{func_class} has positional or keyword args \
                which will not be passed from config."
            )
            continue
        func_args.add(name)

    # Config class arguments.
    if not is_dataclass(config_class):
        raise TypeError(
            f"config_class must be a dataclass, got {type(config_class)}."
        )
    config_args = {f.name for f in fields(config_class)}
    
    unknown_args = config_args - func_args
    missing_args = func_args - config_args
    if unknown_args or missing_args:
        err_msg = f"'{func_class.__name__}' and '{config_class.__name__}' \
            have mismatch arguments:"
        if missing_args:
            err_msg += f"\nmissing args in '{func_class.__name__}': \
                {list(missing_args)}"
        if unknown_args:
            err_msg += f"\nunknown args in '{config_class.__name__}': \
                {list(unknown_args)}"
        raise ValueError(err_msg)


class Registry:
    """A registry class to register pipeline modules such as model, dataset,
    optimizer, schedule, etc.

    Args:
        - module_name: name of pipeline module.
    """
    def __init__(self, module_name: str = "Module") -> None:
        self.module_name = module_name
        self._map: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func_class: Callable[..., Any], 
                 config_class: _type.ConfigClass) -> None:
        if name in self._map:
            raise KeyError(f"{self.module_name} '{name}' already registered.")
        # NOTE: this step ensures class and config have matched arguments,
        # but we do not check argument types.
        _check_arg_match(func_class, config_class)
        self._map[name] = func_class

    def build(self, name: str, config: _type.Config) -> _type.Schedule:
        if not name in self._map:
            raise KeyError(f"{self.module_name} '{name}' not registered.")
        if not is_dataclass(config):
            raise TypeError(
                f"config_class must be a dataclass, got {type(config)}."
            )
        return self._map[name](**asdict(config))
