"""Models and their LightningModule wrappers."""

from collections import OrderedDict

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from torch import nn

# undocumented public methods ok - nn.Module and L.Lightning methods don't need explanation
# ruff: noqa: D102

class CNN(nn.Module):
    """Simple 3 layer CNN with ReLU activations, global average pooling, then 2 fc layers."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cnn_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=cfg.l1_chan,
                                kernel_size=(3, 3), stride=2, padding='valid')),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=cfg.l1_chan, out_channels=cfg.l2_chan,
                                kernel_size=(3, 3), padding='same')),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=cfg.l2_chan, out_channels=cfg.l3_chan,
                                kernel_size=(3, 3), stride=2, padding='valid')),
            ('relu3', nn.ReLU()),
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
        ]))
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(cfg.l3_chan, cfg.fc1_out)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(cfg.fc1_out, 10)),
        ]))

    def forward(self, x):
        x = self.cnn_stack(x)
        x = torch.squeeze(x)
        logits = self.fc_stack(x)

        return logits


class LitCNN(L.LightningModule):
    """Wrap torch.nn model with Lightning."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CNN(cfg)
        # used by ModelSummary to track tensor dimensions through network
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        # give names to each layer in network for activation forward hook
        for name, module in self.named_modules():
            module.name = name
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
  
    @staticmethod
    def tb_hook(logger, step):
        """Tensorboard activation histogram hook."""
        def hook(module, input, output):
            logger.add_histogram('act/' + str(module.name), output, global_step=step)

        return hook
    
    @staticmethod
    def wandb_hook(logger, step):
        """Weights and Biases activation histogram hook."""
        def hook(module, input, output):
            logger.log({'act/' + str(module.name): output}, step=step)

        return hook

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)  # calls forward
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        if self.global_step % 250 == 0:
            tb_logger = self.loggers[0].experiment
            wandb_logger = self.loggers[1].experiment
            num_correct = (logits.detach().argmax(dim=1) == y).sum().type(torch.float).item()
            self.log("train/acc", num_correct/len(y))
            # log weights and gradients
            for n, t in self.named_parameters():
                tb_logger.add_histogram("param/" + n, t.detach(), global_step=self.global_step)
                wandb_logger.log({"param/" + n: wandb.Histogram(t.detach())}, step=self.global_step)
                if t.grad is not None:
                    tb_logger.add_histogram("grad/" + n, t.grad.detach(),
                                            global_step=self.global_step)
                    wandb_logger.log({"grad/" + n: wandb.Histogram(t.grad.detach())},
                                     step=self.global_step)
            with torch.nn.modules.module.register_module_forward_hook(
                LitCNN.tb_hook(tb_logger, self.global_step)):
                self(x)  # run forward pass on current batch with fwd hook enabled
            with torch.nn.modules.module.register_module_forward_hook(
                LitCNN.wandb_hook(wandb_logger, self.global_step)):
                self(x)  # run forward pass on current batch with fwd hook enabled
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)  # calls forward
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        num_correct = (logits.detach().argmax(dim=1) == y).sum().type(torch.float).item()
        self.log("val_acc", num_correct / len(y), prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        if self.cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), **self.cfg.optimizer_params)
        elif self.cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.cfg.optimizer_params)
        return optimizer
