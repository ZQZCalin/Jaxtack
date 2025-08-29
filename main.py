#!/usr/bin/env python3
"""
File:        main.py
Author:      
Created:     
Description: Main Entry of Jaxstack

Notes:
    <write me>
"""

import hydra
from omegaconf import DictConfig

from utils import TimeKeeper
from src import pipeline_init
from src import configurator_init
from src import wandb_init
from src import train_loop_init

# NOTE: set verbose=0 to print no config; 
# verbose=1 to print the final config; 
# verbose=2 to print both initial and final config.
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    
    # step 1: initialize configuration
    # fool-proof preprocessing and sanity checks
    config = configurator_init(config, verbose=2)
    
    # step 2: initialize pipeline
    # all components in the pipeline can be customized in their corresponding directories
    train_state, optimizer, train_loader, loss_fn, logger, wandb_logger = pipeline_init(config)

    # step 3: initialize wandb (weights and bias)
    wandb_init(config)
    
    # step 4: initialize train loop, from the components in the pipeline
    train_loop_init(
        config = config,
        train_state = train_state,
        optimizer = optimizer,
        dataloader = train_loader,
        loss_fn = loss_fn,
        logger = logger,
        time_keeper = TimeKeeper(),
        wandb_logger = wandb_logger,
    )

if __name__ == "__main__":
    main()