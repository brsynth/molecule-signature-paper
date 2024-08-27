#!/usr/bin/env python
import argparse
from pathlib import Path
import logging
import math
import sys
from types import SimpleNamespace
from typing import List, Tuple
import yaml

import coloredlogs
import sentencepiece as spm
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

from paper.dataset.utils import log_config
from paper.transformer.model import Transformer


# Logging -----------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if logger.hasHandlers():  # Jupyter Notebooks
    logger.handlers.clear()

coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Jupyter Notebooks
)

logger.setLevel(logging.INFO)


# Handy functions -----------------------------------------------------------------------------

def dict_to_simplenamespace(d) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_simplenamespace(v)
        else:
            d[k] = v
    return SimpleNamespace(**d)


def namespace_to_dict(obj):
    from types import SimpleNamespace
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj


def load_config(file: str) -> SimpleNamespace:
    with open(file, "r") as file:
        return dict_to_simplenamespace(yaml.safe_load(file))


def show_config(config):
    from pprint import pprint
    pprint(namespace_to_dict(config), sort_dicts=False)


# Training functions ---------------------------------------------------------------------------

class TSVDataset_LOWMEM(Dataset):
    def __init__(
            self,
            file_path: str | Path,
            col_idx: tuple[int, int],
            sp_models: tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor],
            max_length: int = None,
            bos_idx: int = 1,
            eos_idx: int = 2,
    ):
        self.file_path = file_path
        self.col_indexes = {
            "source": col_idx[0],
            "target": col_idx[1],
        }
        self.sp_models = {
            "source": sp_models[0],
            "target": sp_models[1],
        }
        self.offsets = self._index_file()
        self.max_length = max_length
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def _index_file(self) -> List[int]:
        offsets = []
        with open(self.file_path, 'r') as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _trim_sequence(self, tokens: List[int]) -> List[int]:
        if self.max_length is not None:
            return tokens[:self.max_length]
        return tokens

    def _add_beos_indexes(self, tokens: List[int]) -> List[int]:
        return [self.bos_idx] + tokens + [self.eos_idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self.offsets[idx]
        with open(self.file_path, 'r') as f:
            f.seek(offset)  # Go straight to the line
            line = f.readline().strip().split('\t')
            source = line[self.col_indexes["source"]]
            target = line[self.col_indexes["target"]]

            # Tokenization
            source_tokens = self.sp_models["source"].encode_as_ids(source)
            target_tokens = self.sp_models["target"].encode_as_ids(target)

            # Adding BOS and EOS tokens
            source_tokens = self._add_beos_indexes(source_tokens)
            target_tokens = self._add_beos_indexes(target_tokens)

            # Trimming
            source_tokens = self._trim_sequence(source_tokens)
            target_tokens = self._trim_sequence(target_tokens)

            return torch.tensor(source_tokens), torch.tensor(target_tokens)

    def __len__(self):
        return len(self.offsets)


class TSVDataset(Dataset):
    def __init__(
            self,
            file_path: str | Path,
            col_idx: tuple[int, int],
            sp_models: tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor],
            seq_lengths: tuple[int, int] = (100, None),
            max_length: int = None,
            bos_idx: int = 1,
            eos_idx: int = 2,
    ):
        self.file_path = file_path
        self.col_indexes = {
            "source": col_idx[0],
            "target": col_idx[1],
        }
        self.max_length = {
            "source": seq_lengths[0],
            "target": seq_lengths[1],
        }
        self.sp_models = {
            "source": sp_models[0],
            "target": sp_models[1],
        }
        self.data = self._load_data()
        if self.max_length["source"] is None:
            self.max_length["source"] = self._find_longuest_sequence("source")
        else:
            self.max_length["source"] = max_length[0]
        if self.max_length["target"] is None:
            self.max_length["target"] = self._find_longuest_sequence("target")
        else:
            self.max_lentgh["target"] = max_length[1]
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def _load_data(self) -> List[Tuple[str, str]]:
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                source = line[self.col_indexes["source"]]
                target = line[self.col_indexes["target"]]
                data.append((source, target))
        return data

    def _find_longuest_sequence(self, mode: str, percent: int = 95) -> int:
        from numpy import percentile
        if mode not in ["source", "target"]:
            raise ValueError("mode must be either 'source' or 'target'")
        if mode == "source":
            val = percentile([len(sequence) for sequence, _ in self.data], percent)
            return int(val)
        else:
            val = percentile([len(sequence) for _, sequence in self.data], percent)
            return int(val)

    def _trim_sequence(self, tokens: List[int], max_length: int = None) -> List[int]:
        if max_length is not None:
            return tokens[:max_length]
        return tokens

    def _add_beos_indexes(self, tokens: List[int]) -> List[int]:
        return [self.bos_idx] + tokens + [self.eos_idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source, target = self.data[idx]

        # Tokenization
        source_tokens = self.sp_models["source"].encode_as_ids(source)
        target_tokens = self.sp_models["target"].encode_as_ids(target)

        # Adding BOS and EOS tokens
        source_tokens = self._add_beos_indexes(source_tokens)
        target_tokens = self._add_beos_indexes(target_tokens)

        # Trimming
        source_tokens = self._trim_sequence(source_tokens, self.max_length["source"])
        target_tokens = self._trim_sequence(target_tokens, self.max_length["target"])

        return torch.tensor(source_tokens), torch.tensor(target_tokens)

    def __len__(self):
        return len(self.data)


def load_sp_models(
        CONFIG: SimpleNamespace,
        spm_dir: str | Path = None,
) -> Tuple[
    spm.SentencePieceProcessor,
    spm.SentencePieceProcessor
]:
    # Models directory
    spm_dir = Path(spm_dir)
    # Source model
    source_file = Path(spm_dir) / f"{CONFIG.source}.model"
    source_model = spm.SentencePieceProcessor(source_file.as_posix())
    # Target model
    target_file = Path(spm_dir) / f"{CONFIG.target}.model"
    target_model = spm.SentencePieceProcessor(target_file.as_posix())

    return source_model, target_model


def set_up_model(CONFIG) -> torch.nn.Module:
    # Set up model
    model = Transformer(
        d_model=CONFIG.model.dim_model,
        nhead=CONFIG.model.num_heads,
        num_encoder_layers=CONFIG.model.num_encoder_layers,
        num_decoder_layers=CONFIG.model.num_decoder_layers,
        dropout=CONFIG.model.dropout_rate,
        src_vocab_size=getattr(CONFIG.tokenizer, CONFIG.source).vocab_size,
        tgt_vocab_size=getattr(CONFIG.tokenizer, CONFIG.target).vocab_size,
    )
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_normal_(p)
            # nn.init.xavier_uniform_(p)

    return model


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """Generate a square subsequent mask for a sequence.

    Generates a square mask for a sequence, where the mask is a lower triangular
    matrix with values below the diagonal set to `True` and values on or above
    the diagonal set to `False`.

    Parameters
    ----------
    size : int
        The size of the square mask (both the number of rows and columns).

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape (size, size) containing the square subsequent mask.
    """
    return torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)


def create_collate_fn(pad_idx: int):
    """Create a pre-parametrized collate function for DataLoader."""

    def collate_fn(batch):
        """Collate function for DataLoader that pads and masks sequences.

        Parameters
        ----------
        batch : list
            List of tuples containing source and target sequences.

        Returns
        -------
        src : torch.Tensor
            Source sequences.
        tgt_input : torch.Tensor
            Target sequences without the last token.
        tgt_output : torch.Tensor
            Target sequences without the first token.
        tgt_mask : torch.Tensor
            Mask for the target input sequences.
        src_padding_mask : torch.Tensor
            Padding mask for the source sequences.
        tgt_padding_mask : torch.Tensor
            Padding mask for the target input sequences.
        """
        src, tgt = zip(*batch)

        # Padding
        src = pad_sequence(src, batch_first=True, padding_value=pad_idx)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_idx)

        # Shift target to force autoregressive behavior
        tgt_input = tgt[:, :-1]  # Remove last token
        tgt_output = tgt[:, 1:]  # Remove first token

        # Masking
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))  # 2D tensor (tgt_len, tgt_len)  # noqa
        src_padding_mask = (src == pad_idx)  # 2D tensor (batch_size, src_len)
        tgt_padding_mask = (tgt_input == pad_idx)  # 2D tensor (batch_size, tgt_len)

        return src, tgt_input, tgt_output, tgt_mask, src_padding_mask, tgt_padding_mask

    return collate_fn


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        log_interval: int = 100,
        logger: logging.Logger = logging.getLogger(__name__),
) -> float:
    """Train the model on the training data for one epoch.

    Notice: the model is in place, so no need to return it.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    criterion : torch.nn.Module
        Loss function to be used.
    data_loader : DataLoader
        DataLoader containing training data.
    device : torch.device
        Torch device.

    Returns
    -------
    float
        Average loss over the training data.
    """
    model.train()
    total_loss = 0.
    model.to(device)

    # Initialiser le GradScaler pour gérer les gradients en mixed precision
    scaler = torch.GradScaler(device.type)

    for batch_idx, batch in enumerate(data_loader):

        (
            source,
            target_input,
            target_output,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask
        ) = batch

        source = source.to(device)
        target_input = target_input.to(device)
        target_output = target_output.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Activate mixed precision
        with torch.autocast(device.type):

            # Pass data through the model
            output = model(
                source,
                target_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Calculate loss
            try:
                loss = criterion(output.view(-1, output.size(-1)), target_output.view(-1))
            except RuntimeError:
                # view() is not supported for "cpu" torch.device
                loss = criterion(output.reshape(-1, output.size(-1)), target_output.reshape(-1))

        # Continue with scaling
        scaler.scale(loss).backward()  # Back-propagate
        scaler.step(optimizer)  # Update weights
        scaler.update()  # Update the scale

        # Legacy code
        # loss.backward()  # Back-propagate
        # optimizer.step()  # Update weights

        # Update learning rate
        scheduler.step()  # Update learning rate

        # Log loss
        total_loss += loss.item()  # Log loss

        # Log progress
        if batch_idx != 0 and batch_idx % log_interval == 0:
            logger.info(f"  L Batch {batch_idx:>5} / {len(data_loader)} - Loss: {loss.item():.4f}")

    # Log loss for the whole epoch
    avg_loss = total_loss / (batch_idx + 1)

    # Return the average loss
    return avg_loss


def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate the model on the validation data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    criterion : torch.nn.Module
        Loss function to be used.
    data_loader : DataLoader
        DataLoader containing validation data.
    device : torch.device
        Torch device.

    Returns
    -------
    avg_loss : float
        Average loss over the validation data.
    accuracy : float
        Accuracy over the validation data.
    perplexity : float
        Perplexity over the validation data.
    BLEU : float
        BLEU score over the validation data (not implemented ATM).

    """
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_predictions = 0
    model.to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            (
                source,
                target_input,
                target_output,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask
            ) = batch

            source = source.to(device)
            target_input = target_input.to(device)
            target_output = target_output.to(device)
            tgt_mask = tgt_mask.to(device)
            src_padding_mask = src_padding_mask.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            # Pass data through the model
            output = model(
                source,
                target_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_output.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            pred_tokens = output.argmax(dim=-1)
            correct_predictions = (pred_tokens == target_output).sum().item()
            total_correct += correct_predictions
            total_predictions += target_output.numel()

    avg_loss = total_loss / (batch_idx + 1)  # Get the average loss
    accuracy = total_correct / total_predictions  # Get the accuracy
    perplexity = math.exp(avg_loss)  # Get the perplexity

    return avg_loss, accuracy, perplexity


def save_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        loss: float,
        save_dir: str | Path,
        model_name: str = "model.pt",
):
    """Save the model, optimizer, epoch and loss to a file.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler to save.
    epoch : int
        Epoch to save.
    loss : float
        Loss to save.
    save_dir : str | Path
        Directory where to save the model.
    model_name : str
        Name of the model file.
    """
    # Create the directory if it does not exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build up the model path
    model_path = save_dir / model_name

    # Save the model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        model_path,
    )

    # Log
    logger.debug(f"Model saved at {model_path}")


def load_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        load_path: str | Path,
        device: torch.device,
) -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
    int,
    float,
]:
    """Load the model, optimizer, epoch and loss from a file.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load.
    optimizer : torch.optim.Optimizer
        Optimizer to load.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler to load.
    load_path : str
        Name of the model file.
    device : torch.device
        Torch device to load the model to.

    Returns
    -------
    model : torch.nn.Module
        Loaded model.
    optimizer : torch.optim.Optimizer
        Loaded optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Loaded scheduler.
    epoch : int
        Loaded epoch.
    loss : float
        Loaded loss.
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logger.debug(f"Model loaded from {load_path}")

    return model, optimizer, scheduler, epoch, loss


# Main ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name, description="Train a Transformer model"
    )
    parser.add_argument(
        "--source",
        metavar="TYPE",
        type=str,
        choices=["SMILES", "SIGNATURE", "ECFP"],
        default="SMILES",
        help="Source type (default: %(default)s; choices: {%(choices)s})",
    )
    parser.add_argument(
        "--target",
        metavar="TYPE",
        type=str,
        choices=["SMILES", "SIGNATURE", "ECFP"],
        default="SIGNATURE",
        help="Target type (default: %(default)s); choices: {%(choices)s})",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        default="metanetx",
        choices=("metanetx", "emolecules"),
        help="Dataset to tokenize (default: %(default)s; choices: %(choices)s)",
    )
    parser.add_argument(
        "--base_path",
        metavar="DIR",
        type=str,
        default="data/metanetx",
        help="Base path for the dataset tree (default: %(default)s)",
    )
    parser.add_argument(
        "--model_dir",
        metavar="DIR",
        type=str,
        default="models",
        help="Directory name to save the models (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default=Path(__file__).parent / "config.yaml",
        help="Configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="Device to use (default: %(default)s)",
    )

    args = parser.parse_args()

    # Load configuration ------------------------
    CONFIG = load_config(args.config)
    for k, v in vars(args).items():
        setattr(CONFIG, k, v)
    log_config(CONFIG, logger=logger)

    # Set base path -----------------------------
    base_path = CONFIG.base_path

    # Set seed ----------------------------------
    torch.manual_seed(CONFIG.training.seed)

    # SentencePiece models ----------------------
    spm_dir = Path(base_path) / "spm"
    src_sp_model, tgt_sp_model = load_sp_models(CONFIG, spm_dir)

    # Set device --------------------------------
    CONFIG.device = torch.device(CONFIG.device)

    # Load dataset ----------------------------------------------------------------
    data_path = Path(base_path) / CONFIG.data.dataset_file

    # Unzip if necessary
    if data_path.suffix == ".gz":
        if data_path.with_suffix("").exists():
            logger.debug("Data file already unzipped")
            data_path = data_path.with_suffix("")
        else:
            logger.debug("Unzipping data file ...")
            import gzip
            import shutil
            with gzip.open(data_path, 'rb') as f_in:
                with open(data_path.with_suffix(""), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    # Load as torch.Dataset
    src_col_idx = vars(CONFIG.tokenizer)[CONFIG.source].col_idx
    tgt_col_idx = vars(CONFIG.tokenizer)[CONFIG.target].col_idx

    dataset = TSVDataset(
        file_path=data_path,
        col_idx=(src_col_idx, tgt_col_idx),
        sp_models=(src_sp_model, tgt_sp_model),
        max_length=(None, None),  # Will be set to 95th percentile
        bos_idx=CONFIG.tokenizer.BOS_IDX,
        eos_idx=CONFIG.tokenizer.EOS_IDX,
    )

    # Debug values ------------------------------
    DATASET_LEN = len(dataset)
    DATASET_LEN = 1000
    CONFIG.training.kfold = 2
    CONFIG.training.batch_size = 2
    CONFIG.training.epochs = 10
    CONFIG.training.log_interval = 1
    NB_LOADER_WORKERS = 0

    # K-Fold -----------------------------------
    n_splits = CONFIG.training.kfold
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CONFIG.training.seed
    )

    # Training loop -----------------------------
    nb_batches = DATASET_LEN // CONFIG.training.batch_size
    model_dir = Path(base_path) / CONFIG.model_dir
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(DATASET_LEN))):

        # Log fold
        logger.info(f"Fold {fold + 1:>2}/{n_splits}")

        # Split the dataset into training and validation sets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Set up dataloaders
        collate_fn = create_collate_fn(pad_idx=CONFIG.tokenizer.PAD_IDX)
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=CONFIG.training.batch_size,
            shuffle=False,  # Already shuffled by KFold
            collate_fn=collate_fn,
            num_workers=NB_LOADER_WORKERS,
            pin_memory=True,  # Expected to speed up training
        )
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=CONFIG.training.batch_size,
            shuffle=False,  # No need to shuffle
            collate_fn=collate_fn,
            num_workers=NB_LOADER_WORKERS,
            pin_memory=True,  # Expected to speed up training
        )

        # Set up a fresh model at each fold
        model = set_up_model(CONFIG)
        if torch.cuda.is_available():
            model = torch.compile(model)  # Expected to speed up training
            logger.debug("  L Model compiled")
        else:
            logger.debug("  L Torch did not compile the model")

        # Set up optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG.training.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=CONFIG.training.learning_rate / 100,
            max_lr=CONFIG.training.learning_rate,
            step_size_up=nb_batches // 2,  # One cycle per epoch
        )

        # Set up loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Loop over epochs
        for epoch in range(CONFIG.training.epochs):

            # Log
            logger.info(f"Epoch {epoch + 1:>3}")

            # Train the model
            avg_loss = train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                data_loader=train_loader,
                device=CONFIG.device,
                log_interval=CONFIG.training.log_interval,
            )

            # Evaluate the model
            val_loss, val_accuracy, val_perplexity = evaluate(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                device=CONFIG.device,
            )

            # Log the evaluation metrics
            logger.info(
                f"  L Epoch {epoch + 1:>3} - "
                f"    L Training Loss: {avg_loss:.4f} - "
                f"    L Validation Loss: {val_loss:.4f} - "
                f"    L Validation Accuracy: {val_accuracy:.4f} - "
                f"    L Validation Perplexity: {val_perplexity:.4f}"
            )

            # Save states
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                save_dir=model_dir / f"fold_{fold+1:02}",
                model_name=f"model_epoch_{epoch+1:03}.pt",
            )
