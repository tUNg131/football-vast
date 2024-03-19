import os
from datetime import datetime
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformer import TrainableFootballTransformer

def main(hparams: Namespace) -> None:
    seed_everything(hparams.seed, workers=True)

    # Init model and data
    model = TrainableFootballTransformer(hparams)

    # Init early stopping callback
    early_stop_call_back = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # Init loggers
    tb_logger = TensorBoardLogger(
        save_dir="log/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    ckpt_path = os.path.join(
        "log/",
        tb_logger.version,
        "checkpoints",
    )

    # Init model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="{epoch}-{val_loss:.6f}",
        verbose=True,
        save_weights_only=True,
        monitor=hparams.monitor,
        save_top_k=hparams.save_top_k,
        mode=hparams.metric_mode,
    )

    # Init trainer
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[
            early_stop_call_back,
            checkpoint_callback
        ],
        deterministic=hparams.deterministic,
        check_val_every_n_epoch=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        precision=hparams.precision,
        profiler=hparams.profiler,
        benchmark=hparams.benchmark,
    )

    # Start training
    trainer.fit(model=model, datamodule=model.datamodule)


if __name__ == "__main__":
    #####################
    # Training arguments
    #####################

    parser = ArgumentParser(
        description="Football Transformer",
        add_help=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed."
    )
    parser.add_argument(
        "--save-top-k",
        default=1,
        type=int,
        help=(
            "The best k models according to the"
            " quantity monitored will be saved."
        )
    )

    ######################
    # Early stopping
    ######################
    parser.add_argument(
        "--monitor",
        default="val_loss",
        type=str,
        help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric-mode",
        default="min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=4,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )

    #####################
    # Trainer
    #####################
    parser.add_argument(
        "--deterministic",
        nargs="?",
        const=True,
        default=False,
        help="Enable deterministic run."
    )
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        type=int,
        const=True,
        default=False,
        help=("Enable fast development run.")
    )
    parser.add_argument(
        "--min-epochs",
        default=1,
        type=int,
        help="Limits training to a minimum # of epochs.",
    )
    parser.add_argument(
        "--max-epochs",
        default=40,
        type=int,
        help="Limits training to a max # of epochs",
    )
    parser.add_argument(
        "--precision",
        default="32-true",
        type=str,
        help=(
            "32-true, 16-mixed, bf16-mixed,"
            " transformer-engine, 16-true, bf16-true, 64-true"
        )
    )
    parser.add_argument(
        "--profiler",
        default=None,
        type=str,
        help=(
            "To profile individual steps during training and "
            "assist in identifying bottlenecks."
        ),
        choices=["simple", "advanced"]
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        default=None,
        const=True,
        help="CUDNN auto-tuner will try to find the best algorithm for the hardware."
    )

    TrainableFootballTransformer.update_parser_with_model_args(parser)
    hparams = parser.parse_args()

    ######################
    # Run training
    ######################
    main(hparams)