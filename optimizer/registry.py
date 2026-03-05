# Optimizer registry.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type
from .config import BaseOptimizerConfig


@dataclass
class OptimizerEntry:
    config_cls: Type[BaseOptimizerConfig]
    init_fn: Callable[..., Any] | None = None


class OptimizerRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, OptimizerEntry] = {}

    def register(
        self,
        name: str,
        config_cls: Type[BaseOptimizerConfig],
        init_fn: Callable[..., Any] | None = None,
    ):
        """
        Case 1: default init_fn is the decorated function
            @OPTIMIZER_REGISTRY.register("adam", AdamConfig)
            def init_adam(config: AdamConfig, *args): 
                ...

        Case 2: use a custom function as init_fn
            def init_custom(config: CustomConfig, *args):
                ...
            OPTIMIZER_REGISTRY.register("custom", CustomConfig, init_fn=init_custom)
        """
        if name in self._entries:
            raise ValueError(f"optimizer '{name}' already registered")

        entry = OptimizerEntry(config_cls=config_cls, init_fn=init_fn)
        self._entries[name] = entry

        # Case 1: default init_fn to the decorated function
        if init_fn is None:
            def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
                entry.init_fn = obj
                return obj
            return decorator
        # Case 2: custom init_fn
        else:
            return init_fn

    def get_config_cls(self, name: str) -> Type[BaseOptimizerConfig]:
        try:
            return self._entries[name].config_cls
        except KeyError:
            raise KeyError(f"unknown optimizer '{name}'")

    def init(self, config: BaseOptimizerConfig, **kwargs: Any) -> Any:
        try:
            entry = self._entries[config.name]
        except KeyError:
            raise KeyError(f"unknown optimizer '{config.name}'")

        if entry.init_fn is None:
            raise RuntimeError(
                f"optimizer '{config.name}' has no init_fn registered "
                "(did you forget to decorate or pass init_fn=... ?)"
            )

        return entry.init_fn(config, **kwargs)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())


OPTIMIZER_REGISTRY = OptimizerRegistry()