import inspect
from dataclasses import dataclass, is_dataclass, fields
import hydra
from omegaconf import OmegaConf, DictConfig
from common.registry import _check_arg_match


@dataclass
class Boo:
    x: int
    y: int = 2

def foo(a, b: int, c=3, *, d=4, **kwargs):
    pass

class MyClass:
    def __init__(self, x, y=10, *, z: str = "hi"):
        pass


def test_inspect():
    sig = inspect.signature(foo)
    for name, param in sig.parameters.items():
        print(name, param.kind, param.default)


    print("\n\n")

    sig = inspect.signature(MyClass)
    for name, param in sig.parameters.items():
        print(name, param.kind, param.default)


@dataclass
class FooConfig:
    a: int
    b: int
    c: int
    e: int


def test_check_arg_match():
    _check_arg_match(foo, FooConfig)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))



from schedule import build_schedule
from schedule.config import TrapezoidScheduleConfig

    
def test_schedule_registry():
    name = "trapezoid"
    config = TrapezoidScheduleConfig(
        peak_value=1.0,
        total_steps=8,
        warmup_steps=2,
        decay_steps=2,
    )
    schedule = build_schedule(name, config)
    for i in range(-3, 10):
        print(schedule(i))


if __name__ == "__main__":
    # Boo()
    # test_config()
    test_schedule_registry()
    # test_check_arg_match()

    # test_inspect()

    # config = FooConfig(a=1,b=1,c=1,e=1)
    # print(is_dataclass(FooConfig))
    # print(is_dataclass(config))

    # print(fields(FooConfig) == fields(config))