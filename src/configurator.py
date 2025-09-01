#!/usr/bin/env python3
"""
File:        configurator.py
Author:      
Created:     
Description: <Short summary of what the this does>

Notes:
    <write me>
"""

import inspect
import importlib
from dataclasses import dataclass, is_dataclass
import hydra
from omegaconf import OmegaConf, DictConfig
from optimizer import config as optimizer_config


def configurator_init(config: DictConfig, verbose) -> dataclass:
    pass

