from typing import Any, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from .components.ebay_dataset import eBayDataset


class eBayDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/eBay/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: List[Any] = None,
        val_transforms: List[Any] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose(train_transforms)
        self.val_transforms = transforms.Compose(val_transforms)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes_1(self) -> int:
        return 16

    @property
    def num_classes_2(self) -> int:
        return 75

    @property
    def num_classes_3(self) -> int:
        return 1000

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = eBayDataset("train", self.train_transforms, self.hparams.data_dir)
            self.data_val = eBayDataset("val", self.val_transforms, self.hparams.data_dir)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Test split is the same as val
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
