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


def flatten(d, parent_key='', sep='_'):
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, (int, float, str, bool)):
            items.append((new_key, v))
        else:
            items.extend(flatten(v, new_key, sep=sep).items())

    return dict(items)

class HPMetricCallback(Callback):

    def on_train_end(self, trainer, pl_module):
        tb_logger = pl_module.loggers[0]
        wandb_logger = pl_module.loggers[1]
        hp_params = flatten(pl_module.cfg)
        tb_logger.log_hyperparams(hp_params, trainer.logged_metrics)
        wandb_logger.experiment.config.update(hp_params)



@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    """Training and model evaluation."""
    wandb.init(reinit=True)  # needed with hydra multirun
    print(OmegaConf.to_yaml(cfg))
    cnn = LitCNN(cfg.model)
    mnist_data = MNISTDataModule(cfg.data)
    print(ModelSummary(cnn, max_depth=-1))
    tb_logger = TensorBoardLogger("tb_logs", log_graph=True,
                                  default_hp_metric=False)  # don't log hpparams without metric
    wandb_logger = WandbLogger(project="mnist")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                        patience=3, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss")
    hp_metric_callback = HPMetricCallback()
    trainer = L.Trainer(**cfg.trainer, logger=[tb_logger, wandb_logger],
                        callbacks=[early_stop_callback, checkpoint_callback, hp_metric_callback])
    trainer.fit(cnn, mnist_data)


if __name__ == "__main__":
    main()
