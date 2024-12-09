import argparse
import logging
import os

import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from paper.dataset.utils import setup_logger, log_config
from paper.learning.model import TransformerModel
from paper.learning.config import Config
from paper.learning.utils import Tokenizer, EarlyStoppingCustom


# Main -------------------------------------------------------------------------------------------
def main(CONFIG):

    # Seed
    seed_everything(CONFIG.train_seed)

    # TensorBoard Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=CONFIG.out_dir,
        name=f"{CONFIG.source}-{CONFIG.target}",
        sub_dir=CONFIG.out_sub_dir,
    )

    # Early stopping (custom)
    early_stopping_callback = EarlyStoppingCustom(
        monitor="val/loss",
        mode="min",
        threshold_mode="rel",
        patience=CONFIG.early_stopping_patience,
        min_delta=CONFIG.early_stopping_min_delta,
        verbose=True,
    )

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tensorboard_logger.log_dir, "checkpoints"),  # Use logger directory
        filename=None,  # Use default filename
        monitor="val/loss",
        mode="min",
        save_top_k=20,  # Save top checkpoints
        every_n_epochs=1,  # Save every epoch
        save_last=True,
    )

    # Profiler
    profiler = SimpleProfiler(filename="profiler")

    # Resume training
    if CONFIG.resume or CONFIG.finetune:

        # Checkpoint
        if CONFIG.resume:
            if CONFIG.resume_checkpoint is None:
                raise ValueError("The --resume_checkpoint argument must be set to continue training.")

        else:  # CONFIG.finetune == True
            if CONFIG.finetune_checkpoint is None:
                raise ValueError("The --finetune_checkpoint argument must be set to fine-tune the model.")  # noqa

            else:
                CONFIG.resume_checkpoint = CONFIG.finetune_checkpoint

        # Subset changeable settings
        resume_config = {
            "data_train_file": CONFIG.data_train_file,
            "data_valid_file": CONFIG.data_valid_file,
            "data_col_indexes": (CONFIG.source_col_idx, CONFIG.target_col_idx),
            "data_max_rows": CONFIG.data_max_rows,
            "dataloader_num_workers": CONFIG.dataloader_num_workers,
            "dataloader_seq_max_lengths": (CONFIG.source_max_length, CONFIG.target_max_length),
            "train_batch_size": CONFIG.train_batch_size,
            "train_max_epochs": CONFIG.train_max_epochs,
            "scheduler_method": CONFIG.scheduler_method,
            "scheduler_lr": CONFIG.scheduler_lr,
            "scheduler_exponential_gamma": CONFIG.scheduler_exponential_gamma,
            "scheduler_plateau_patience": CONFIG.scheduler_plateau_patience,
            "scheduler_plateau_factor": CONFIG.scheduler_plateau_factor,
        }

        # Load pre-trained model
        model = TransformerModel.load_from_checkpoint(
            checkpoint_path=CONFIG.resume_checkpoint,
            **resume_config,
        )

        # Fine-tuning
        if CONFIG.finetune:
            # Update learning rate
            if CONFIG.finetune_lr is not None:
                model.scheduler_lr = CONFIG.finetune_lr

            # Freeze encoder
            if CONFIG.finetune_freeze_encoder:
                model.freeze_encoder()

    else:  # CONFIG.resume == False and CONFIG.finetune == False
        # Tokenizers
        source_tokenizer = Tokenizer(CONFIG.source_token_model, fp_type=CONFIG.source)
        target_tokenizer = Tokenizer(CONFIG.target_token_model, fp_type=CONFIG.target)

        # Load fresh model
        model = TransformerModel(
            # Data arguments
            data_train_file=CONFIG.data_train_file,
            data_valid_file=CONFIG.data_valid_file,
            data_col_indexes=(CONFIG.source_col_idx, CONFIG.target_col_idx),
            data_max_rows=CONFIG.data_max_rows,
            # Dataloader arguments
            dataloader_num_workers=CONFIG.dataloader_num_workers,
            dataloader_seq_max_lengths=(CONFIG.source_max_length, CONFIG.target_max_length),
            # Token arguments
            token_fns=(source_tokenizer, target_tokenizer),
            # Model arguments
            model_dim=CONFIG.model_dim,
            model_num_heads=CONFIG.model_num_heads,
            model_num_encoder_layers=CONFIG.model_num_encoder_layers,
            model_num_decoder_layers=CONFIG.model_num_decoder_layers,
            model_dropout=CONFIG.model_dropout,
            # Training arguments
            train_batch_size=CONFIG.train_batch_size,
            train_max_epochs=CONFIG.train_max_epochs,
            # Scheduler arguments
            scheduler_method=CONFIG.scheduler_method,
            scheduler_lr=CONFIG.scheduler_lr,
            scheduler_exponential_gamma=CONFIG.scheduler_exponential_gamma,
            scheduler_plateau_patience=CONFIG.scheduler_plateau_patience,
            scheduler_plateau_factor=CONFIG.scheduler_plateau_factor,
        )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=-1,  # Use all GPUs
        # num_nodes=CONFIG.ddp_num_nodes,
        precision="16-mixed" if CONFIG.enable_mixed_precision else None,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=CONFIG.train_max_epochs,
        val_check_interval=CONFIG.train_val_check_interval,
        enable_progress_bar=True,
        accumulate_grad_batches=CONFIG.train_accumulate_grad_batches,
        profiler=profiler,
    )

    # Train
    trainer.fit(model)


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
setup_logger(logger, logging.INFO)


# Arg parsing -------------------------------------------------------------------------------------
def parser_args():
    # Get default config
    CONFIG = Config(db="metanetx", mode="translate")

    parser = argparse.ArgumentParser()
    parser_general = parser.add_argument_group("General options")
    parser_general.add_argument(
        "--db",
        metavar="DB",
        type=str,
        default=CONFIG.db,
        help="Database to use (default: %(default)s)",
    )
    parser_general.add_argument(
        "--source",
        metavar="TYPE",
        type=str,
        help="Source data type",
        required=True,
    )
    parser_general.add_argument(
        "--target",
        metavar="TYPE",
        type=str,
        help="Target data type",
        required=True,
    )

    # Parallel & computing options
    parser_computing = parser.add_argument_group("Distributed and computing options")
    parser_computing.add_argument(
        "--dataloader_num_workers",
        metavar="WORKERS",
        type=int,
        default=CONFIG.dataloader_num_workers,
        help="Number of workers for each dataloader process. Default: %(default)s.",
    )
    parser_computing.add_argument(
        "--enable_mixed_precision",
        metavar="BOOL",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=CONFIG.enable_mixed_precision,
        help="Whether to enable mixed precision. Default: %(default)s.",
    )
    parser_computing.add_argument(
        "--ddp_num_nodes",
        metavar="NODES",
        type=int,
        help="Number of nodes for distributed training",
    )

    # Data options
    parser_data = parser.add_argument_group("Data options")
    parser_data.add_argument(
        "--split_method",
        metavar="METHOD",
        type=str,
        choices=["kfold", "train_valid_split"],
        default=CONFIG.split_method,
        help=(
            "Data split method used to generate the training and validation files. ",
            "Default: %(default)s. Choices: {%(choices)s}."
        )
    )
    parser_data.add_argument(
        "--train_fold_index",
        metavar="INDEX",
        type=int,
        default=CONFIG.train_fold_index,
        help="Fold index to use (only for 'kfold' split method). Default: %(default)s.",
    )
    parser_data.add_argument(
        "--data_max_rows",
        metavar="N",
        type=int,
        default=CONFIG.data_max_rows,
        help="Maximum number of rows to read from the data file. Default: %(default)s.",
    )

    # Output options
    parser_output = parser.add_argument_group("Output options")
    parser_output.add_argument(
        "--out_dir",
        metavar="DIR",
        type=str,
        default=CONFIG.out_dir,
        help="Output directory for logs and checkpoints. Default: %(default)s.",
    )
    parser_output.add_argument(
        "--out_sub_dir",
        metavar="DIR",
        type=str,
        default=CONFIG.out_sub_dir,
        help="Subdirectory for logs and checkpoints. Default: %(default)s.",
    )

    # Model options
    parser_model = parser.add_argument_group("Model options")
    parser_model.add_argument(
        "--model_dim",
        metavar="DIM",
        type=int,
        default=CONFIG.model_dim,
        help="Model dimension. Default: %(default)s.",
    )
    parser_model.add_argument(
        "--model_num_encoder_layers",
        metavar="LAYERS",
        type=int,
        default=CONFIG.model_num_encoder_layers,
        help="Number of encoder layers. Default: %(default)s.",
    )
    parser_model.add_argument(
        "--model_num_decoder_layers",
        metavar="LAYERS",
        type=int,
        default=CONFIG.model_num_decoder_layers,
        help="Number of decoder layers. Default: %(default)s.",
    )

    # Scheduler options
    parser_scheduler = parser.add_argument_group("Scheduler options")
    parser_scheduler.add_argument(
        "--scheduler_method",
        metavar="METHOD",
        type=str,
        default=CONFIG.scheduler_method,
        help="Scheduler method. Default: %(default)s.",
    )
    parser_scheduler.add_argument(
        "--scheduler_lr",
        metavar="LR",
        type=float,
        default=CONFIG.scheduler_lr,
        help="Learning rate. Default: %(default)s.",
    )
    parser_scheduler.add_argument(
        "--scheduler_exponential_gamma",
        metavar="GAMMA",
        type=float,
        default=CONFIG.scheduler_exponential_gamma,
        help="Exponential learning rate gamma. Default: %(default)s.",
    )
    parser_scheduler.add_argument(
        "--scheduler_plateau_patience",
        metavar="PATIENCE",
        type=int,
        default=CONFIG.scheduler_plateau_patience,
        help="Plateau scheduler patience. Default: %(default)s.",
    )
    parser_scheduler.add_argument(
        "--scheduler_plateau_factor",
        metavar="FACTOR",
        type=float,
        default=CONFIG.scheduler_plateau_factor,
        help="Plateau scheduler factor. Default: %(default)s.",
    )

    # Early stopping options
    parser_stopping = parser.add_argument_group("Early stopping options")
    parser_stopping.add_argument(
        "--early_stopping_patience",
        metavar="PATIENCE",
        type=int,
        default=CONFIG.early_stopping_patience,
        help="Early stopping patience. Default: %(default)s.",
    )
    parser_stopping.add_argument(
        "--early_stopping_min_delta",
        metavar="DELTA",
        type=float,
        default=CONFIG.early_stopping_min_delta,
        help="Early stopping minimum delta. Default: %(default)s.",
    )

    # Training options
    parser_train = parser.add_argument_group("Training options")
    parser_train.add_argument(
        "--train_max_epochs",
        metavar="EPOCHS",
        type=int,
        default=CONFIG.train_max_epochs,
        help="Maximum number of epochs to train. Default: %(default)s.",
    )
    parser_train.add_argument(
        "--train_batch_size",
        metavar="SIZE",
        type=int,
        default=CONFIG.train_batch_size,
        help="Batch size. Default: %(default)s.",
    )
    parser_train.add_argument(
        "--train_accumulate_grad_batches",
        metavar="NUM",
        type=int,
        default=CONFIG.train_accumulate_grad_batches,
        help="Number of batches to accumulate gradients before doing a backward pass. Default: %(default)s.",  # noqa
    )
    parser_train.add_argument(
        "--train_val_check_interval",
        metavar="INTERVAL",
        type=float,
        default=CONFIG.train_val_check_interval,
        help="Validation check interval. Default: %(default)s.",
    )
    parser_train.add_argument(
        "--train_seed",
        metavar="SEED",
        type=int,
        default=CONFIG.train_seed,
        help="Seed for random number generators. Default: %(default)s.",
    )

    # Fine-tuning options
    parser_finetune = parser.add_argument_group("Fine-tuning options")
    parser_finetune.add_argument(
        "--finetune",
        metavar="BOOL",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help=(
            "Whether to fine-tune the model. If True, the model is loaded from the checkpoint "
            "defined by --finetune_checkpoint. Change the --db argument to switch to another set "
            "of train and validation files. Default: %(default)s."
        ),
    )
    parser_finetune.add_argument(
        "--finetune_lr",
        metavar="LR",
        type=float,
        default=None,
        help=(
            "Fine-tuning learning rate. If not set, the learning rate from --scheduler_lr is "
            "used. Default: %(default)s."
        ),
    )
    parser_finetune.add_argument(
        "--finetune_freeze_encoder",
        metavar="BOOL",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Whether to freeze the encoder weights. Default: %(default)s.",
    )
    parser_finetune.add_argument(
        "--finetune_checkpoint",
        metavar="FILE",
        type=str,
        help="Checkpoint file from which to fine-tune the model.",
    )

    # Resume options
    parser_resume = parser.add_argument_group("Resume options")
    parser_resume.add_argument(
        "--resume",
        metavar="BOOL",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help=(
            "Whether to resume training. If True, the model will be loaded from the checkpoint "
            "defined in --resume_checkpoint. Default: False."
        ),
    )
    parser_resume.add_argument(
        "--resume_checkpoint",
        metavar="FILE",
        type=str,
        help="Checkpoint file to resume training.",
    )

    # # Trick to use the command pickargs from VSCODE for debugging
    # if os.getenv("USED_VSCODE_COMMAND_PICKARGS", "0") == "1":
    #     parser.set_defaults(dataset="emolecules")
    #     parser.set_defaults(descriptors_resume=True)
    #     parser.set_defaults(show_progress=True)

    args = parser.parse_args()

    # Update config according to CLI args
    CONFIG = (
        CONFIG
        .with_db(args.db)
        .with_split_method(args.split_method)
        .with_fold_index(args.train_fold_index)
        .with_source(args.source)
        .with_target(args.target)
    )
    for key, value in vars(args).items():
        if value is not None:
            setattr(CONFIG, key, value)

    log_config(CONFIG, logger)

    return CONFIG


# Main call ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse arguments and load configuration
    CONFIG = parser_args()

    # Run
    main(CONFIG)
