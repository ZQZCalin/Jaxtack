# src/initialize.py
"""Initialize: read config from yaml and initialize config classes."""

from pydantic import BaseModel
from omegaconf import OmegaConf, DictConfig
from _optimizer import build_optimizer


def initialize(cfg: DictConfig) -> DictConfig:
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    optimizer_cfg = cfg.optimizer
    trainer_cfg = cfg.trainer

    model = build_model(model_cfg)
    dataset = build_dataset(dataset_cfg)
    optimizer = build_optimizer(optimizer_cfg)
    trainer = build_trainer(
        trainer_cfg,
        model = model,
        dataset = dataset,
        optimizer = optimizer,
    )

    