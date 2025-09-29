import os
import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import emulator.src.utils.config_utils as cfg_utils
from emulator.src.utils.interface import get_model_and_data
from emulator.src.utils.utils import get_logger
from pytorch_lightning.profilers import PyTorchProfiler
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler, schedule
from codecarbon import EmissionsTracker
import torch



def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = get_logger(__name__)
    emissions_tracker_enabled = config.get('datamodule', {}).get('emissions_tracker', False)
    log.info("In run model")
    cfg_utils.extras(config)

    log.info("Running model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")


    emulator_model, data_module = get_model_and_data(config)

    log.info(f"Got model - {config.name}")
    profiler = None
    checkpointing = True
        
    log.info(config.name)
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        profiler=profiler,
        callbacks=callbacks,
        logger=loggers,  # , deterministic=True
        enable_checkpointing=checkpointing,
    )

    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    cfg_utils.log_hyperparameters(
        config=config,
        model=emulator_model,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
    )
    resume_ckpt = config.ckpt_path or None
    trainer.fit(model=emulator_model, datamodule=data_module, ckpt_path=resume_ckpt)
    trainer.test(datamodule=data_module, ckpt_path="best")
    
