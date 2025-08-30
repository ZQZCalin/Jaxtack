import inspect
from dataclasses import dataclass
import hydra
from omegaconf import OmegaConf, DictConfig


@dataclass
class Boo:
    x: int
    y: int = 2

def foo(a, b: int, c=3, *args, d=4, **kwargs):
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



@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    # Boo()
    test_config()

    # test_inspect()