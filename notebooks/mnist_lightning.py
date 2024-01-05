import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import torch
from torch import nn
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter  # tensorboard
import wandb  # weights and biases

from torchsummary import summary

p = Path('.') / 'data' / 'MNIST'
train_data = datasets.MNIST(str(p), train=True, download=True, transform=ToTensor())
train_data, val_data = random_split(train_data, lengths=[0.9, 0.1])
test_data = datasets.MNIST(str(p), train=False, download=True, transform=ToTensor())
                                     
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential accepts OrderedDict if you'd like to give layers meaningful names
        self.cnn_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=2, padding='valid')),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding='same')),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding='valid')),
            ('relu3', nn.ReLU()),
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
        ]))
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32, 32)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(32, 10)),
        ]))
        # Give each module a useful name for tracking activations in
        # TensorBoard and Weights & Biases
        for name, module in self.named_modules():
            module.name = name

    def forward(self, x):
        x = self.cnn_stack(x)
        x = torch.squeeze(x)
        logits = self.fc_stack(x)

        return logits

class LitCNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()   
        self.model = model
        # Give each module a useful name for tracking activations in
        # TensorBoard and Weights & Biases
        for name, module in self.named_modules():
            module.name = name

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        return optimizer

cnn = LitCNN(CNN())
trainer = L.Trainer(max_epochs=5, enable_progress_bar=True)
trainer.fit(cnn, train_dataloader, val_dataloader)