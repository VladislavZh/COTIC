import os
from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase as Logger
from pytorch_lightning.callbacks import DeviceStatsMonitor

from src import utils

log = utils.get_logger(__name__)

data_train = None
data_val = None
data_test = None
normalizer = None


def train(config: DictConfig) -> Optional[float]:
    global data_train
    global data_val
    global data_test
    global normalizer
    try:
        """Contains the training pipeline. Can additionally evaluate model on a testset, using best
        weights achieved during training.
    
        Args:
            config (DictConfig): Configuration composed by Hydra.
    
        Returns:
            Optional[float]: Metric score for hyperparameter optimization.
        """

        # Set seed for random number generators in pytorch, numpy and python.random
        if config.get("seed"):
            seed_everything(config.seed, workers=True)

        # Convert relative ckpt path to absolute path if necessary
        ckpt_path = config.trainer.get("resume_from_checkpoint")
        if ckpt_path and not os.path.isabs(ckpt_path):
            config.trainer.resume_from_checkpoint = os.path.join(
                hydra.utils.get_original_cwd(), ckpt_path
            )

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        datamodule.data_train = data_train
        datamodule.data_val = data_val
        datamodule.data_test = data_test
        if normalizer is not None:
            datamodule.normalizer = normalizer

        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Init lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init lightning loggers
        logger: List[Logger] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Train the model
        if config.get("train"):
            log.info("Starting training!")
            if config.get("resume_from_checkpoint") is not None:
                print("Resuming from ckpt.")
                trainer.fit(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=config.get("resume_from_checkpoint")
                )
            else:
                trainer.fit(model=model, datamodule=datamodule)

        data_train = datamodule.data_train
        data_val = datamodule.data_val
        data_test = datamodule.data_test
        normalizer = datamodule.normalizer

        # Get metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric and optimized_metric not in trainer.callback_metrics:
            raise Exception(
                "Metric for hyperparameter optimization not found! "
                "Make sure the `optimized_metric` in `hparams_search` config is correct!"
            )
        score = trainer.callback_metrics.get(optimized_metric)

        # Test the model
        if config.get("test"):
            # ckpt_path = "best"
            # if not config.get("train") or config.trainer.get("fast_dev_run"):
            #     ckpt_path = None
            log.info("Starting testing!")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path)

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Print path to best checkpoint
        if not config.trainer.get("fast_dev_run") and config.get("train"):
            log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        return score

    except torch.cuda.OutOfMemoryError:
        return -1000
