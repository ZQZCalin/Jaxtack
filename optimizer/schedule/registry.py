# Learning rate schedule registry.

from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any, Callable, Dict, Type
from optax import ScalarOrSchedule
from omegaconf import OmegaConf, DictConfig


class BaseLearningRateConfig(BaseModel):
    name: str


@dataclass
class LearningRateEntry:
    config_cls: Type[BaseLearningRateConfig]
    builder: Callable[..., Any] | None = None


class LearningRateRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, LearningRateEntry] = {}

    def register(
        self,
        name: str,
        config_class: Type[BaseLearningRateConfig],
        builder: Callable[..., Any] | None = None,
    ):
        if name in self._entries:
            raise ValueError(f"learning rate schedule '{name}' already registered")

        entry = LearningRateEntry(config_cls=config_class, builder=builder)
        self._entries[name] = entry

        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            entry.builder = obj
            return obj
        return decorator

    def build(self, config: BaseLearningRateConfig) -> ScalarOrSchedule:
        try:
            entry = self._entries[config.name]
        except KeyError:
            raise KeyError(f"unknown learning rate schedule '{config.name}'")

        if entry.builder is None:
            raise RuntimeError(
                f"learning rate schedule '{config.name}' has no builder registered "
                "(did you forget to decorate or pass builder=... ?)"
            )

        return entry.builder(config)

    def get_config_class(self, name: str) -> Type[BaseLearningRateConfig]:
        try:
            return self._entries[name].config_cls
        except KeyError:
            raise KeyError(f"unknown learning rate schedule '{name}'")

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())


LEARNING_RATE_REGISTRY = LearningRateRegistry()


def build_learning_rate(cfg: DictConfig) -> ScalarOrSchedule:
    name: str = cfg.name
    ConfigClass = LEARNING_RATE_REGISTRY.get_config_class(name)
    learning_rate_config = ConfigClass(**OmegaConf.to_container(cfg, resolve=True))
    return LEARNING_RATE_REGISTRY.build(learning_rate_config)
