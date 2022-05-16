from typing import Any, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms

from .components.ebay_dataset import eBayDataset, eBayRetrievalDataset


class eBayDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/eBay/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: List[Any] = None,
        val_transforms: List[Any] = None,
        aug_transforms: List[Any] = None,
        query_key: str = "query_part2",
        retrieval_setting: bool = False,
        index_trainable: bool = False,
        multi_label: bool = False,
        concat_train_index: bool = False,
        load_train_for_predict: bool = False,
        load_query_only: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose(train_transforms)
        self.val_transforms = transforms.Compose(val_transforms)
        if aug_transforms is not None:
            self.aug_transforms = transforms.Compose(aug_transforms)
        else:
            self.aug_transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_index: Optional[Dataset] = None
        self.data_query: Optional[Dataset] = None
        self.data_train_for_predict: Optional[Dataset] = None

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
            if self.hparams.retrieval_setting:
                if self.hparams.index_trainable:
                    self.data_train = eBayRetrievalDataset(
                        "index",
                        self.train_transforms,
                        self.hparams.data_dir,
                    )
                else:
                    self.data_train = eBayRetrievalDataset(
                        "train",
                        self.train_transforms,
                        self.hparams.data_dir,
                    )
            elif self.hparams.concat_train_index:
                data_train = eBayDataset(
                    "train",
                    self.train_transforms,
                    self.hparams.data_dir,
                    self.aug_transforms,
                    self.hparams.multi_label,
                )
                data_index = eBayDataset(
                    "index",
                    self.train_transforms,
                    self.hparams.data_dir,
                    self.aug_transforms,
                    self.hparams.multi_label,
                )
                self.data_train = ConcatDataset([data_train, data_index])
            else:
                self.data_train = eBayDataset(
                    "train",
                    self.train_transforms,
                    self.hparams.data_dir,
                    self.aug_transforms,
                    self.hparams.multi_label,
                )
            self.data_val = eBayDataset(
                "val", self.val_transforms, self.hparams.data_dir, None, self.hparams.multi_label
            )
        if not self.data_index and not self.data_query:
            self.data_index = eBayDataset("index", self.val_transforms, self.hparams.data_dir)
            self.data_query = eBayDataset(
                self.hparams.query_key, self.val_transforms, self.hparams.data_dir
            )
            if self.hparams.load_train_for_predict:
                self.data_train_for_predict = eBayDataset(
                    "train", self.val_transforms, self.hparams.data_dir
                )

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

    def predict_dataloader(self):
        query_loader = DataLoader(
            dataset=self.data_query,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        # FIXME: temp fix to handle query2 only
        if self.hparams.load_query_only:
            return query_loader, query_loader
        index_loader = DataLoader(
            dataset=self.data_index,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        if self.hparams.load_train_for_predict:
            train_loader = DataLoader(
                dataset=self.data_train_for_predict,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
            return query_loader, index_loader, train_loader
        return query_loader, index_loader
