from typing import Optional, List

import os

import torch
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.datamodules.components.BaseKFoldDataModule import BaseKFoldDataModule

label_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

class FBDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 is_train: bool = True,
                 ):
        self.df = df
        self.full_text = self.df['full_text'].values

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.is_train = is_train

        if self.is_train:
            self.label = self.df[label_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        full_text = self.full_text[index]
        label = self.label[index]

        inputs = self.tokenizer(full_text, max_length=self.max_length, padding='max_length', truncation=True)

        if self.is_train:
            return torch.tensor(inputs['input_ids'], dtype=torch.long), \
                   torch.tensor(inputs['attention_mask'], dtype=torch.long), \
                   torch.tensor(label, dtype=torch.long)

        return torch.tensor(inputs['input_ids'], dtype=torch.long), \
               torch.tensor(inputs['attention_mask'], dtype=torch.long)


class FBDataModule(BaseKFoldDataModule):
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        num_class: int,
        max_length: int,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super(FBDataModule, self).__init__()

        self.save_hyperparameters()

        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.current_train_dataset: Optional[Dataset] = None
        self.current_valid_dataset: Optional[Dataset] = None

        self.splits: Optional[List] = None

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.hparams.model_dir)

    @property
    def num_class(self):
        return self.hparams.num_class

    def setup(self, stage: Optional[str] = None) -> None:
        self.df_train = pd.read_csv(os.path.join(self.hparams.data_dir, 'train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.hparams.data_dir, 'test.csv'))

        self.train_dataset = FBDataset(self.df_train, self.tokenizer, max_length=self.hparams.max_length)
        self.test_dataset = FBDataset(self.df_test, self.tokenizer, max_length=self.hparams.max_length, is_train=False)

    def setup_folds(self, num_folds: int) -> None:
        self.splits = [
            split for split in MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True).
            split(self.df_train, self.df_train[label_cols])
        ]

    def setup_fold_index(self, current_fold: int):
        train_indices, valid_indices = self.splits[current_fold]
        self.current_train_dataset = Subset(self.train_dataset, train_indices)
        self.current_valid_dataset = Subset(self.train_dataset, valid_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.current_train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.current_valid_dataset,
            batch_size=self.hparams.valid_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "fb.yaml")
    cfg.data_dir = os.getenv("DATASET_PATH")
    cfg.model_dir = os.getenv("PRETRAINED_MODEL_PATH")
    _ = hydra.utils.instantiate(cfg)
