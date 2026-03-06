# Optimizer registry.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type
from optax import GradientTransformation
from .config import BaseOptimizerConfig


@dataclass
class OptimizerEntry:
    config_cls: Type[BaseOptimizerConfig]
    builder: Callable[..., Any] | None = None


class OptimizerRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, OptimizerEntry] = {}

    def register(
        self,
        name: str,
        config_cls: Type[BaseOptimizerConfig],
        builder: Callable[..., Any] | None = None,
    ):
        """
        Case 1: default builder is the decorated function
            @OPTIMIZER_REGISTRY.register("adam", AdamConfig)
            def build_adam(config: AdamConfig, *args): 
                ...

        Case 2: use a custom function as builder
            def build_custom(config: CustomConfig, *args):
                ...
            OPTIMIZER_REGISTRY.register("custom", CustomConfig, builder=build_custom)
        """
        if name in self._entries:
            raise ValueError(f"optimizer '{name}' already registered")

        entry = OptimizerEntry(config_cls=config_cls, builder=builder)
        self._entries[name] = entry

        # Case 1: default builder to the decorated function
        if builder is None:
            def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
                entry.builder = obj
                return obj
            return decorator
        # Case 2: custom builder
        else:
            return builder

    def build(self, config: BaseOptimizerConfig, **kwargs: Any) -> GradientTransformation:
        try:
            entry = self._entries[config.name]
        except KeyError:
            raise KeyError(f"unknown optimizer '{config.name}'")
        if entry.builder is None:
            raise RuntimeError(
                f"optimizer '{config.name}' has no init_fn registered "
                "(did you forget to decorate or pass init_fn=... ?)"
            )
        return entry.builder(config, **kwargs)

    def get_config_class(self, name: str) -> Type[BaseOptimizerConfig]:
        try:
            return self._entries[name].config_cls
        except KeyError:
            raise KeyError(f"unknown optimizer '{name}'")

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())


OPTIMIZER_REGISTRY = OptimizerRegistry()