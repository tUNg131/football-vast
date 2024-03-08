import argparse

def h5(root, save_path, ball_delta_t, joints_delta_t, sequence_length):
    import os

    import numpy as np
    from tqdm import tqdm

    from data import save_to_h5, data_generator

    gen = tqdm(data_generator(root, ball_delta_t, joints_delta_t, sequence_length))
    data = np.fromiter(gen, dtype=np.dtype((float, (sequence_length, 30, 3))))
    np.random.shuffle(data)

    # Split to train, validation, test dataset
    split1 = int(0.7 * len(data))
    split2 = int(0.9 * len(data))
    train = data[:split1]
    val = data[split1:split2]
    test = data[split2:]

    save_to_h5(train, os.path.join(save_path, "train.hdf5"))
    save_to_h5(val, os.path.join(save_path, "val.hdf5"))
    save_to_h5(test, os.path.join(save_path, "test.hdf5"))


def train(train_path, val_path, batch_size, max_epochs, version, fast_dev_run, precision,
          random, model_args, model_kwargs):
    import os

    import torch

    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    from model import HumanPoseModel
    from dataset import HumanPoseDataset

    # Set up
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    num_workers = os.cpu_count() - 1
    train_loader = DataLoader(HumanPoseDataset(train_path, drop_type=random),
                              batch_size=batch_size,
                              num_workers=num_workers)
    val_loader = DataLoader(HumanPoseDataset(val_path, drop_type=random),
                            batch_size=batch_size, num_workers=num_workers)

    logger = TensorBoardLogger("tb_logs", version=version)
    trainer = Trainer(logger=logger,
                      precision=precision,
                      max_epochs=max_epochs,
                      fast_dev_run=fast_dev_run)

    model = HumanPoseModel(*model_args, **model_kwargs)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train model")

    subparsers = parser.add_subparsers(help='train, h5', dest="command")

    # Subparser for save data to H5 files
    parser_h5 = subparsers.add_parser("h5", help="Save data to H5 files")
    parser_h5.add_argument("--src", "-S", type=str, required=True, help="Raw data directory")
    parser_h5.add_argument("--dst", "-D", type=str, help="Directory to store H5 files", default="")
    parser_h5.add_argument("--bdt", "-B", type=int, required=True, help="Ball time difference")
    parser_h5.add_argument("--jdt", "-J", type=int, required=True, help="Joint time difference")
    parser_h5.add_argument("--len", "-L", type=int, required=True, help="Sequence length")

    # Subparse for train model
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--train-path", type=str, required=True)
    parser_train.add_argument("--val-path", type=str, required=True)
    parser_train.add_argument("--batch-size", type=int, default=128)
    parser_train.add_argument("--version", type=str, required=True)
    parser_train.add_argument("--max-epochs", type=int, required=True)
    parser_train.add_argument("--random", type=str, required=True)
    parser_train.add_argument("--precision", type=str, required=True,
        help="32-true, 16-mixed, bf16-mixed, transformer-engine, 16-true, bf16-true, 64-true")
    parser_train.add_argument("--fast-dev-run", nargs='?', const=True, type=int, default=False,
        help='Enable fast development run. Optionally, specify a value.')

    parser.add_argument("--n-timestep", type=int, default=32, help="Number of timesteps")
    parser.add_argument("--n-joint", type=int, default=15, help="Number of joints")
    parser.add_argument("--d-joint", type=int, default=3, help="Dimension of each joint")
    parser.add_argument("--d-x", type=int, default=3, help="Dimension of x")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--d-model", type=int, default=1024, help="Dimension of the model")
    parser.add_argument("--d-hid", type=int, default=2048, help="Dimension of the hidden layer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    args = parser.parse_args()

    if args.command == "h5":
        h5(args.src, args.dst, args.bdt, args.jdt, args.len)
        print("Save data to: ", args.dst)
    elif args.command == "train":
        model_args = (
            args.n_timestep,
            args.n_joint,
            args.d_joint,
            args.d_x,
            args.n_heads,
            args.n_layers,
            args.d_model,
            args.d_hid,
            args.dropout
        )
        model_kwargs = {}

        train(train_path=args.train_path,
              val_path=args.val_path,
              batch_size=args.batch_size,
              version=args.version,
              max_epochs=args.max_epochs,
              fast_dev_run=args.fast_dev_run,
              precision=args.precision,
              random=args.random,
              model_args=model_args,
              model_kwargs=model_kwargs)
    else:
        print("Invalid command. Use --help for available commands.")