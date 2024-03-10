import os
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Type, Optional, Any

import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import optimizer
from dataset import (
    HumanPoseDataset,
    SetOriginToCenterOfMassAndAlignXY,
    DropRandomChunkVariableSize,
    DropRandomUniform,
    DropRandomChunk,
    AddRandomNoise
)


DEFAULT_LOADER_WORKER_COUNT = os.cpu_count() - 1


class Embedding(nn.Module):

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.linear = nn.Linear(
            in_features=hparams.d_joint,
            out_features=hparams.d_model
        )

        self.time_embedding = nn.Embedding(
            num_embeddings=hparams.n_timesteps,
            embedding_dim=hparams.d_model
        )

        self.joint_embedding = nn.Embedding(
            num_embeddings=hparams.n_joints,
            embedding_dim=hparams.d_model
        )

        self.nan_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=hparams.d_model
        )

        timestep_labels = (
            torch.arange(hparams.n_timesteps, dtype=torch.int)
            .view(1, -1, 1)
            .repeat(hparams.batch_size, 1, hparams.n_joints)
            .view(hparams.batch_size, -1)
        )
        self.register_buffer("timestep_labels", timestep_labels)

        joint_labels = (
            torch.arange(hparams.n_joints, dtype=torch.int)
            .repeat(hparams.batch_size, hparams.n_timesteps)
        )
        self.register_buffer("joint_labels", joint_labels)

    def forward(self, x):
        time_emb = self.time_embedding(self.timestep_labels)

        joint_emb = self.joint_embedding(self.joint_labels)

        nan_emb = self.nan_embedding(
            torch.isnan(x)
            .view(self.hparams.batch_size, -1, self.hparams.d_joint)
            .any(-1)
            .int()
        )

        val_emb = self.linear(
            torch.nan_to_num(x)
            .view(self.hparams.batch_size, -1, self.hparams.d_joint)
        )

        return val_emb + time_emb + joint_emb + nan_emb


class FootballTransformer(pl.LightningModule):

    class DataModule(pl.LightningDataModule):

        def __init__(self, model: Type[pl.LightningModule]) -> None:
            super().__init__()
            self.save_hyperparameters(model.hparams)
            self.model = model

        def get_loader_kwargs(self, dataset, training: Optional[bool] = False) -> dict:
            return {
                "dataset": dataset,
                "shuffle": training,
                "batch_size": self.hparams.batch_size,
                "num_workers": self.hparams.loader_workers,
                "pin_memory": self.model.on_gpu,
                "drop_last": True,
            }

        def get_masking_strategy_args(self, strategy: Any) -> Tuple[int]:
            return (int(strategy[-2:]),)

        def get_transform(self, training: bool = False) -> List[callable]:
            transform = []
            # Align dataset
            if self.hparams.align_dataset:
                transform.append(SetOriginToCenterOfMassAndAlignXY())

            # Training noise
            if training and self.hparams.training_noise_std > 0:
                transform.append(AddRandomNoise(self.hparams.training_noise_std))

            # Mask transform
            masking_strategy = self.hparams.masking_strategy
            args = self.get_masking_strategy_args(masking_strategy)
            if masking_strategy.startswith("vchunk"):
                masking_transform_cls = DropRandomChunkVariableSize
            elif masking_strategy.startswith("chunk"):
                masking_transform_cls = DropRandomChunk
            elif masking_strategy.startswith("random"):
                masking_transform_cls = DropRandomUniform
            else:
                masking_transform_cls = DropRandomChunkVariableSize

            transform.append(masking_transform_cls(*args))
            return transform

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            self._train_dataset = HumanPoseDataset(
                path=self.hparams.train_path,
                n_timesteps=self.hparams.n_timesteps,
                transform=self.get_transform(training=True),
            )
            loader_kwargs = self.get_loader_kwargs(dataset=self._train_dataset,
                                                   training=True)
            return DataLoader(**loader_kwargs)

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            self._val_dataset = HumanPoseDataset(
                path=self.hparams.val_path,
                n_timesteps=self.hparams.n_timesteps,
                transform=self.get_transform(training=False),
            )
            loader_kwargs = self.get_loader_kwargs(dataset=self._val_dataset)
            return DataLoader(**loader_kwargs)

        def test_dataloader(self) -> DataLoader:
            """Function that loads the test set."""
            self._test_dataset = HumanPoseDataset(
                path=self.hparams.test_path,
                n_timesteps=self.hparams.n_timesteps,
                transform=self.get_transform(training=False),
            )
            loader_kwargs = self.get_loader_kwargs(dataset=self._test_dataset)
            return DataLoader(**loader_kwargs)

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        # Build Data module
        self.datamodule = self.DataModule(self)

        # Build model
        self.__build_model()

        # Build loss criterion
        self.__build_loss()

    def __build_model(self) -> None:
        """Init the model."""
        self.embedding = Embedding(self.hparams)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hparams.d_model,
                nhead=self.hparams.n_heads,
                dim_feedforward=self.hparams.d_feedforward,
                dropout=self.hparams.dropout,
                activation=self.hparams.activation,
                batch_first=True
            ),
            num_layers=self.hparams.n_layers,
        )

        self.linear = nn.Linear(
            in_features=self.hparams.d_model,
            out_features=self.hparams.d_joint
        )

    def __build_loss(self) -> None:
        """Initializes the loss function/s."""
        self._loss = nn.functional.mse_loss

    def __forward(self, x: Tensor) -> Tensor:
        embeddings = self.embedding(x)
        features = self.encoder(embeddings)
        output = self.linear(features).view(x.shape)
        return output

    def loss(self, inputs: Tensor, model_out: Tensor, targets: Tensor) -> Tensor:
        nan_mask = torch.isnan(inputs)
        return self._loss(model_out[nan_mask], targets[nan_mask])

    def training_step(self,
                      batch: Tuple[Tensor, Tensor],
                      batch_index: int) -> Tensor:
        
        inputs, targets = batch
        model_out = self.__forward(inputs)
        loss = self.loss(inputs, model_out, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self,
                        batch: Tuple[Tensor, Tensor],
                        batch_index: int) -> Tensor:
        
        inputs, targets = batch
        model_out = self.__forward(inputs)
        loss =  self.loss(inputs, model_out, targets)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer_name = self.hparams.optimizer
        try:
            optimizer_class = getattr(optimizer, optimizer_name)
        except AttributeError:
            optimizer_class = getattr(torch.optim, optimizer_name)

        optimizer_obj = optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer_obj

    @classmethod
    def update_parser_with_model_args(cls: Type[pl.LightningModule],
                                      parser: ArgumentParser) -> None:
        parser.add_argument(
            "--n-timesteps",
            default=32,
            type=int,
            help="# of timesteps in the input."
        )
        parser.add_argument(
            "--n-joints",
            default=14,
            type=int,
            help="# of joints in one timestep."
        )
        parser.add_argument(
            "--d-joint",
            default=3,
            type=int,
            help="Dimension of one joint."
        )
        parser.add_argument(
            "--n-heads",
            default=8,
            type=int,
            help="# of attention heads."
        )
        parser.add_argument(
            "--n-layers",
            default=8,
            type=int,
            help="# of layers."
        )
        parser.add_argument(
            "--d-model",
            default=1024,
            type=int,
            help="Dimension of the model (embedding size)."
        )
        parser.add_argument(
            "--d-feedforward",
            default=2048,
            type=int,
            help="Dimension of the feedforward layer."
        )
        parser.add_argument(
            "--dropout",
            default=0.2,
            type=float,
            help="Dropout probability."
        )
        parser.add_argument(
            "--activation",
            default="relu",
            type=str,
            help="The activation function of the layers"
        )
        parser.add_argument(
            "--learning-rate",
            default=0.0015,
            type=float,
            help="Model learning rate."
        )
        parser.add_argument(
            "--loader-workers",
            default=DEFAULT_LOADER_WORKER_COUNT,
            type=int,
            help=(
                "How many subprocesses to use for data loading."
                " 0 means that the data will be loaded in the main process."
            )
        )
        parser.add_argument(
            "--optimizer",
            default="LAMB",
            type=str,
            help="Name of the optimizer. Can be inside torch.optim."
        )
        parser.add_argument(
            "--train-path",
            default="data/train.hdf5",
            type=str,
            help="Path to the file containing the training data."
        )
        parser.add_argument(
            "--val-path",
            default="data/val.hdf5",
            type=str,
            help="Path to the file containing the validation data."
        )
        parser.add_argument(
            "--test-path",
            default="data/test.hdf5",
            type=str,
            help="Path to the file containing the test data."
        )
        parser.add_argument(
            "--batch-size",
            default=128,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--align-dataset",
            const=True,
            nargs="?",
            default=False,
            help="Recenter & rotate data."
        )
        parser.add_argument(
            "--training-noise-std",
            default=0.02,
            type=float,
            help="Training noise standard deviation."
        )
        parser.add_argument(
            "--masking-strategy",
            default="vchunk15",
            type=str,
            help="Mask strategy. e.g. vchunk15, chunk15, random50.",
        )