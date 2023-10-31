import os
import time

import pytorch_lightning as pl
import torchvision
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder, CIFAR10

from .data_config import DATA_ROOT, IMAGENET_PATH
from .sampler import RASampler
from .transforms import RandomMixup, RandomCutmix

__all__ = ["ImageNetDataModule", "CIFAR10DataModule", "MyBaseDataModule"]


class MyBaseDataModule(pl.LightningDataModule):
    NUM_CLASSES: int = None
    NUM_TRAIN_EXAMPLES: int = None
    NUM_EVAL_EXAMPLES: int = None

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(mixup_alpha, cutmix_alpha)

    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        shuffle = None if train_sampler is not None else True
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def get_train_collate_fn(cls, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
        collate_fn = None
        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler


class ImageNetDataModule(MyBaseDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 1_281_167
    NUM_EVAL_EXAMPLES: int = 50_000

    def setup(self, stage: str) -> None:
        if stage == "fit":
            rank_zero_info("Setting up ImageNet train dataset...")
            start = time.perf_counter()
            self.train_dataset = ImageFolder(
                root=os.path.join(IMAGENET_PATH, "train"),
                transform=self.config["train_transform"],
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNet val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, "val"),
            transform=self.config["test_transform"],
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class CIFAR10DataModule(MyBaseDataModule):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    NUM_CLASSES: int = 10

    NUM_TRAIN_EXAMPLES: int = 50_000
    NUM_EVAL_EXAMPLES: int = 10_000

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = CIFAR10(
                root=DATA_ROOT,
                train=True,
                transform=self.config["train_transform"],
                download=True,
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES

        self.eval_dataset = CIFAR10(
            root=DATA_ROOT,
            train=False,
            transform=self.config["test_transform"],
            download=True,
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
