import os
from argparse import ArgumentParser, Namespace
from typing import Tuple, Type, Optional, Literal
from collections import defaultdict

import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import optimizer
from model import FootballTransformer
from dataset import (
    HumanPoseMidHipDatasetWithGeometricInvariantFeatures,
    DropRandomChunkVariableSize,
    DropRandomUniform,
    DropRandomChunk,
    DEFAULT_JOINT_INDICES
)


DEFAULT_LOADER_WORKER_COUNT = os.cpu_count() - 1


class TrainableFootballTransformer(pl.LightningModule):

    class DataModule(pl.LightningDataModule):
        __dataset_class__  = HumanPoseMidHipDatasetWithGeometricInvariantFeatures

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
                self._train_dataset = self.__dataset_class__(
                    path=self.hparams.train_path,
                    drop=drop,
                    noise=self.hparams.training_noise_std,
                )

                self._val_dataset = self.__dataset_class__(
                    path=self.hparams.val_path,
                    drop=drop,
                )

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            return DataLoader(dataset=self._train_dataset,
                              shuffle=True,
                              batch_size=self.hparams.train_batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            return DataLoader(dataset=self._val_dataset,
                              shuffle=False,
                              batch_size=self.hparams.val_batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def get_test_dataloader(self, gap_size) -> DataLoader:
            dataset = self.__dataset_class__(
                path=self.hparams.test_path,
                drop=DropRandomChunk(gap_size),
            )
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=self.hparams.val_batch_size,
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
        return self.__predict_all(x)
    
    def __nan_mask(self, x: Tensor) -> Tensor:
        hparams = self.hparams
        return (
            x
            .isnan()
            .view(-1, hparams.n_timesteps, hparams.n_joints * hparams.d_joint)
            .any(dim=(2,))
        )
    
    def __nan_start_end_timesteps(self, x: Tensor) -> Tensor:
        hparams = self.hparams

        # Find positions of NaN values
        nan_positions = self.__nan_mask(x).float()

        # Find the first and last NaN positions
        start_nan = nan_positions.argmax(dim=1)
        end_nan = hparams.n_timesteps - nan_positions.flip(dims=(1,)).argmax(dim=1) - 1

        return start_nan, end_nan
    
    def __predict_all(self, x: Tensor) -> Tensor:
        # Predict all timesteps
        return self._model(x)
    
    def __predict_interpolate(self, x: Tensor) -> Tensor:
        # Predict all timesteps using interpolation
        device = self.device
        batch_size, *_ = x.shape
        out = x.clone()

        s, e = self.__nan_start_end_timesteps(x)
        missing_timesteps_count = e[0] - s[0] + 1
        weights = (
            torch
            .linspace(0, 1, missing_timesteps_count + 2, device=device)
            [1:-1]
            .view(-1, 1, 1, 1)
            .repeat(1, batch_size, self.hparams.n_joints, self.hparams.d_joint)
        )

        a = torch.arange(batch_size)
        b = a.unsqueeze(-1)
        c = torch.stack(
            [torch.arange(s_, e_ + 1) for s_, e_ in zip(s, e)],
            dim=0
        )
        
        interpolated = (
            torch
            .lerp(out[a, s - 1], out[a, e + 1], weights)
            .transpose(0, 1)
        )
        out[b, c] = interpolated
        
        return out
    
    def __predict_rollout(self,
                          x: Tensor,
                          strategy: Optional[Literal["both", "left", "right"]] = "both"
                          ) -> Tensor:
        batch_size, *_ = x.shape
        out = x.clone()

        # Check if all batches have the same number of missing timesteps
        missing_timesteps_count = self.__nan_mask(x).sum(dim=1)
        if not torch.all(missing_timesteps_count == missing_timesteps_count[0]):
            raise RuntimeError("Samples in the batch have different # of missing timesteps.")
        
        # Iterate until all NaN values are replaced
        while out.isnan().any():
            tmp = self._model(out).to(out.dtype)

            s, e = self.__nan_start_end_timesteps(out)

            # Replace NaN values with corresponding output values
            i = torch.arange(batch_size)
            if strategy == "left":
                out[i, s] = tmp[i, s]
            elif strategy == "right":
                out[i, e] = tmp[i, e]
            elif strategy == "both":
                out[i, s] = tmp[i, s]
                out[i, e] = tmp[i, e]
            
        return out

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
        self.loss_labels = ["All", "Outside in", "Left to right", "Right to left", "Interpolation"]

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

        inputs, targets = batch
        batch_size, *_ = inputs.shape

        nan_position = torch.isnan(inputs)

        # Compute outputs
        model_output = torch.stack([
            self.__predict_all(inputs),
            self.__predict_rollout(inputs, "both"),
            self.__predict_rollout(inputs, "left"),
            self.__predict_rollout(inputs, "right"),
            self.__predict_interpolate(inputs)
        ])

        # Compute loss for NaN positions
        strategy_count = model_output.size(0)
        o = model_output[:, nan_position]
        t = targets[nan_position].repeat(strategy_count, 1)

        model_loss = (
            self
            ._loss(o, t, reduction='none')
            .view(strategy_count, batch_size, -1, self.hparams.n_joints, self.hparams.d_joint)
        )

        model_losses.append(model_loss)

    def on_test_epoch_end(self) -> None:
        """
        Performs actions at the end of each test epoch.
        """

        for model_loss in self.model_losses.values():
            np_model_loss_avg = (
                torch.cat(model_loss, dim=1)
                .mean(dim=(1, 3, 4), keepdim=False)
                .cpu()
                .numpy()
            )

            missing_timesteps_count = np_model_loss_avg.shape[1]
            x = np.arange(missing_timesteps_count)
            width = 0.15
            multiplier = 0

            plt.figure(figsize=(10, 20))

            for label, loss in zip(self.loss_labels, np_model_loss_avg):
                offset = width * multiplier

                rects = plt.barh(x + offset, loss, width, label=label)
                plt.bar_label(rects, padding=3)

                multiplier += 1

            plt.xlabel("MSE")
            plt.ylim(-1, 15)
            plt.ylabel("Timesteps")
            plt.gca().invert_yaxis()
            plt.yticks(x + width * len(self.loss_labels) / 2, x)
            plt.title(f"MSE by strategies (gap size = {missing_timesteps_count})")
            plt.legend(loc="upper right",
                       ncols=len(self.loss_labels))
            
            # Add the plot to TensorBoard
            self.logger.experiment.add_figure(f"Loss/gapsize = {missing_timesteps_count}", plt.gcf())

        # Clear the test losses for the next epoch
        self.model_losses.clear()
        self.loss_labels.clear()
 
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
            "--train-batch-size",
            default=168,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--val-batch-size",
            default=1536,
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