#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    print("...")

if __name__ == "__main__":
    main()