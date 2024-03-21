import os
from argparse import ArgumentParser, Namespace
from typing import Tuple, Type, Optional
from collections import defaultdict

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import optimizer
from model import FootballTransformer
from dataset import (
    HumanPoseMidHipDataset,
    DropRandomChunkVariableSize,
    DropRandomUniform,
    DropRandomChunk,
    DEFAULT_JOINT_INDICES
)


DEFAULT_LOADER_WORKER_COUNT = os.cpu_count() - 1


class TrainableFootballTransformer(pl.LightningModule):

    class DataModule(pl.LightningDataModule):

        def __init__(self, model: Type[pl.LightningModule]) -> None:
            super().__init__()
            self.save_hyperparameters(model.hparams)
            self.model = model

        def setup(self, stage: str) -> None:
            drop_type, *drop_args = self.hparams.masking_strategy
            if drop_type == "vchunk":
                drop = DropRandomChunkVariableSize(*drop_args)
            elif drop_type == "chunk":
                drop = DropRandomChunk(*drop_args)
            elif drop_type == "random":
                drop = DropRandomUniform(*drop_args)
            else:
                raise

            if stage == "fit":
                self._train_dataset = HumanPoseMidHipDataset(
                    path=self.hparams.train_path,
                    drop=drop,
                    noise=self.hparams.training_noise_std,
                )

                self._val_dataset = HumanPoseMidHipDataset(
                    path=self.hparams.val_path,
                    drop=drop,
                )

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            return DataLoader(dataset=self._train_dataset,
                              shuffle=True,
                              batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            return DataLoader(dataset=self._val_dataset,
                              shuffle=False,
                              batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def get_test_dataloader(self, gap_size) -> DataLoader:
            dataset = HumanPoseMidHipDataset(
                path=self.hparams.test_path,
                drop=DropRandomChunk(gap_size),
            )
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
                pin_memory=self.model.on_gpu
            )

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
        self._model = FootballTransformer(n_timesteps=self.hparams.n_timesteps,
                                          n_joints=self.hparams.n_joints,
                                          d_joint=self.hparams.d_joint,
                                          n_heads=self.hparams.n_heads,
                                          n_layers=self.hparams.n_layers,
                                          d_model=self.hparams.d_model,
                                          d_feedforward=self.hparams.d_feedforward,
                                          dropout=self.hparams.dropout,
                                          activation=self.hparams.activation)

    def __build_loss(self) -> None:
        """Initializes the loss function/s."""
        self._loss = nn.functional.mse_loss

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)
    
    def fill_nan_by_interpolating(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = x.view(-1,
                   self.hparams.n_timesteps,
                   self.hparams.n_joints,
                   self.hparams.d_joint)

        nan_mask = torch.flatten(torch.isnan(x), start_dim=2).any(dim=2).float()

        start = nan_mask.argmax(dim=1) - 1
        end = x.size(dim=1) - nan_mask.flip(dims=(1,)).argmax(dim=1)

        for i, (s, e) in enumerate(zip(start, end)):
            gap_size = e - s - 1
            weights = torch.linspace(0, 1, gap_size + 2, device=self.device)[1:-1]
            x[i, s + 1:e] = torch.lerp(x[i, s:s+1].expand(gap_size, -1, -1),
                                       x[i, e:e+1].expand(gap_size, -1, -1),
                                       weights.view(-1, 1, 1))
        
        x = x.view(-1,
                   self.hparams.n_timesteps * self.hparams.n_joints,
                   self.hparams.d_joint)
        return x

    def loss(self, inputs: Tensor, model_out: Tensor, targets: Tensor) -> Tensor:
        nan_position = torch.isnan(inputs)
        return self._loss(model_out[nan_position], targets[nan_position])

    def training_step(self,
                      batch: Tuple[Tensor, Tensor],
                      batch_index: int) -> Tensor:
        
        inputs, targets = batch
        model_out = self._model(inputs)
        loss = self.loss(inputs, model_out, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self,
                        batch: Tuple[Tensor, Tensor],
                        batch_index: int) -> Tensor:
        
        inputs, targets = batch
        model_out = self._model(inputs)
        loss =  self.loss(inputs, model_out, targets)
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def on_test_epoch_start(self) -> None:
        # Initialize test losses at the start of each epoch
        self.model_losses = defaultdict(list)
        self.baseline_losses = defaultdict(list)

    def test_step(self,
                  batch: Tuple[Tensor, Tensor],
                  batch_index: int,
                  dataloader_idx: Optional[int] = 0) -> None:
        """
        Performs a single test step.
        
        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing input and target tensors.
            batch_index (int): Index of the current batch.
        """
        model_losses = self.model_losses[f"d{dataloader_idx}"]
        baseline_losses = self.baseline_losses[f"d{dataloader_idx}"]

        inputs, targets = batch
        batch_size, *_ = inputs.shape

        nan_position = torch.isnan(inputs)
        model_output = self._model(inputs)
        baseline_output = self.fill_nan_by_interpolating(inputs)

        # Compute loss for NaN positions
        model_loss = (
            self
            ._loss(model_output[nan_position], targets[nan_position], reduction='none')
            .view(batch_size, -1, self.hparams.n_joints, self.hparams.d_joint)
        )
        baseline_loss = (
            self
            ._loss(baseline_output[nan_position], targets[nan_position], reduction='none')
            .view(batch_size, -1, self.hparams.n_joints, self.hparams.d_joint)
        )

        model_losses.append(model_loss)
        baseline_losses.append(baseline_loss)

    def on_test_epoch_end(self) -> None:
        """
        Performs actions at the end of each test epoch.
        """

        for key in self.model_losses:
            model_losses = self.model_losses[key]
            baseline_losses = self.baseline_losses[key]

            model_loss_avg = (
                torch.cat(model_losses, dim=0)
                .mean(dim=(0, 2, 3), keepdim=False)
                .cpu()
            )

            baseline_loss_avg = (
                torch.cat(baseline_losses, dim=0)
                .mean(dim=(0, 2, 3), keepdim=False)
                .cpu()
            )

            # Determine the gap size
            total_timesteps = model_loss_avg.size(0)

            timesteps = range(1, total_timesteps + 1)

            # Plotting MSE for gap size
            plt.ylabel("MSE")
            plt.xlabel("Timesteps")
            
            plt.plot(timesteps,
                    model_loss_avg,
                    label=f"Model loss",
                    marker='o',
                    linestyle='none')
            plt.plot(timesteps,
                    baseline_loss_avg,
                    label=f"Interpolation loss",
                    marker='o',
                    linestyle='none')
            
            plt.title(f"MSE for gap size = {total_timesteps}")
            plt.legend()

            # Add the plot to TensorBoard
            self.logger.experiment.add_figure(f"Loss/gapsize = {total_timesteps}", plt.gcf())

        # Clear the test losses for the next epoch
        self.model_losses.clear()
        self.baseline_losses.clear()
 
    def configure_optimizers(self) -> Optimizer:
        optimizer_name = self.hparams.optimizer
        try:
            optimizer_class = getattr(optimizer, optimizer_name)
        except AttributeError:
            optimizer_class = getattr(torch.optim, optimizer_name)

        optimizer_obj = optimizer_class(
            self._model.parameters(),
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
            default=len(DEFAULT_JOINT_INDICES),
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
            default="JITLamb",
            type=str,
            help="Name of the optimizer. Can be inside torch.optim."
        )
        parser.add_argument(
            "--train-path",
            default="data/h5/10fps/train.hdf5",
            type=str,
            help="Path to the file containing the training data."
        )
        parser.add_argument(
            "--val-path",
            default="data/h5/10fps/val.hdf5",
            type=str,
            help="Path to the file containing the validation data."
        )
        parser.add_argument(
            "--test-path",
            default="data/h5/10fps/test.hdf5",
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
            "--training-noise-std",
            default=0.02,
            type=float,
            help="Training noise standard deviation."
        )
        parser.add_argument(
            "--masking-strategy",
            default=["vchunk", 15],
            nargs="+",
            help="Mask strategy. e.g. vchunk15, chunk15, random50.",
        )