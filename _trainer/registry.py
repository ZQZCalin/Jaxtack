# trainer/registry.py
"""Trainer registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type
from trainer.config import BaseTrainerConfig


@dataclass
class TrainerEntry:
    config_cls: Type[BaseTrainerConfig]
    initializer: Callable[..., Any]
    trainer: Callable[..., Any]


class TrainerRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, TrainerEntry] = {}

    def register(
        self,
        name: str,
        config_cls: Type[BaseTrainerConfig],
        builder: Callable[..., Any] | None = None,
    ):
        if name in self._entries:
            raise ValueError(f"trainer '{name}' already registered")

        entry = TrainerEntry(config_cls=config_cls, builder=builder)
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

    def build(self, config: BaseTrainerConfig, **kwargs: Any) -> Any:
        try:
            entry = self._entries[config.name]
        except KeyError:
            raise KeyError(f"unknown trainer '{config.name}'")

        if entry.builder is None:
            raise RuntimeError(
                f"trainer '{config.name}' has no builder registered "
                "(did you forget to decorate or pass builder=... ?)"
            )

        return entry.builder(config, **kwargs)

    def get_config_class(self, name: str) -> Type[BaseTrainerConfig]:
        try:
            return self._entries[name].config_cls
        except KeyError:
            raise KeyError(f"unknown trainer '{name}'")

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())


TRAINER_REGISTRY = TrainerRegistry()