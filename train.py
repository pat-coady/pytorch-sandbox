#!/usr/bin/env python3
"""Main training and evaluation code."""

import hydra
import lightning as L
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from omegaconf import OmegaConf

from data import MNISTDataModule
from models import LitCNN


def flatten(d):
    """Flatten nested dictionary - assumes no duplicate keys."""
    items = []
    # TODO: warn on duplicate keys
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)):
            items.append((k, v))
        else:
            items.extend(flatten(v).items())

    return dict(items)

class HPMetricCallback(Callback):
    """Callback to log hyperparameters and final model performance (i.e., metrics)."""

    def on_train_end(self, trainer, pl_module):
        tb_logger = pl_module.loggers[0]
        hp_params = flatten(pl_module.cfg)
        tb_logger.log_hyperparams(hp_params, trainer.logged_metrics)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    """Training and model evaluation."""
    print(OmegaConf.to_yaml(cfg))
    cnn = LitCNN(cfg.model)
    mnist_data = MNISTDataModule(cfg.data)
    print(ModelSummary(cnn, max_depth=-1))
    # wandb.init(project='mnist', config=flatten(cfg),
    #            dir='logs')
    tb_logger = TensorBoardLogger(save_dir="logs/tb", name='', log_graph=True,
                                  default_hp_metric=False)  # don't log hpparams without metric
    wandb_logger = WandbLogger(save_dir='logs')
    wandb_logger.experiment.config.update(flatten(cfg))
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01,
                                        patience=3, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss",
                                          dirpath='logs/checkpoints')
    hp_metric_callback = HPMetricCallback()
    trainer = L.Trainer(**cfg.trainer, logger=[tb_logger, wandb_logger],
                        callbacks=[early_stop_callback, checkpoint_callback, hp_metric_callback])
    trainer.fit(cnn, mnist_data)
    wandb_logger.finalize('success')
    tb_logger.finalize('success')

if __name__ == "__main__":
    main()
