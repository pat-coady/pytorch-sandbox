"""Datasets, loaders and DataModules."""

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


class MNISTDataModule(L.LightningDataModule):
    """MNIST Data Module."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.data_dir = cfg.data_dir
        self.transform = ToTensor()

    def prepare_data(self):  # this step done on single (main) CPU thread
        datasets.MNIST(str(self.data_dir), train=True, download=True, transform=self.transform)
        datasets.MNIST(str(self.data_dir), train=False, download=True, transform=self.transform)

    def setup(self, stage):  # this step run on every CPU
        if stage == "fit":
            train_data = datasets.MNIST(self.data_dir,
                                        train=True, download=False, transform=ToTensor())
            train_data, val_data = random_split(train_data, lengths=[0.9, 0.1])
            self.train_data = train_data
            self.val_data = val_data
        if stage == "test":
            test_data = datasets.MNIST(self.data_dir,
                                       train=False, download=False, transform=ToTensor())
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          num_workers=self.cfg.num_workers, drop_last=True)
