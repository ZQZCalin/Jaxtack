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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # NOTE: set verbose=0 to print no config; 
    # verbose=1 to print the final config; 
    # verbose=2 to print both initial and final config.
    config = init_config(config, verbose=2)
    
    # NOTE: customize each component in `_src` if needed.
    train_state, optimizer, train_loader, loss_fn, logger, wandb_logger = init_pipeline(config)

    time_keeper = TimeKeeper()

    init_wandb(config)
    
    # NOTE: customize your own train_loop in `_src/train`.
    # train_loop = init_train_loop(config)
    lm_train_loop(
        config = config,
        train_state = train_state,
        optimizer = optimizer,
        dataloader = train_loader,
        loss_fn = loss_fn,
        logger = logger,
        time_keeper = time_keeper,
        wandb_logger = wandb_logger,
    )


if __name__ == "__main__":
    main()