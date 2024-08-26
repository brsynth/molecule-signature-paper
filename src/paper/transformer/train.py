#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import yaml

import coloredlogs
import sentencepiece as spm
import torch
import torch.nn as nn

from paper.dataset.utils import log_config
from paper.transformer.model import Transformer


# Utilities functions -----------------------------------------------------------------------------
def dict_to_simplenamespace(d) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_simplenamespace(v)
        else:
            d[k] = v
    return SimpleNamespace(**d)


def load_config(file: str) -> SimpleNamespace:
    with open(file, "r") as file:
        return dict_to_simplenamespace(yaml.safe_load(file))


def set_up_model(
    CONFIG,
) -> Tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler,
    nn.Module,
]:

    logger.debug("  L Setting up layers...")
    model = Transformer(
        d_model=CONFIG.model.dim_model,
        nhead=CONFIG.model.num_heads,
        num_encoder_layers=CONFIG.model.num_encoder_layers,
        num_decoder_layers=CONFIG.model.num_decoder_layers,
        dropout=CONFIG.model.dropout_rate,
        src_vocab_size=getattr(CONFIG.tokenizer, CONFIG.source).vocab_size,
        tgt_vocab_size=getattr(CONFIG.tokenizer, CONFIG.target).vocab_size,
    )

    # Initialize weights -------------------------
    logger.debug("  L Initializing weights...")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
            # nn.init.xavier_uniform_(p)

    return model


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


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
        help="Source type (default: %(default)s); choices: {%(choices)s}",
    )
    parser.add_argument(
        "--target",
        metavar="TYPE",
        type=str,
        choices=["SMILES", "SIGNATURE, ECFP"],
        default="SIGNATURE",
        help="Target type (default: %(default)s); choices: {%(choices)s}",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        default="metanetx",
        choices=("metanetx", "emolecules"),
        help="Dataset to tokenize (default: %(default)s; choices: %(choices)s)"
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default=Path(__file__).parent / "config.yaml",
        help="Configuration file (default: %(default)s)",
    )

    args = parser.parse_args()

    # Load configuration ------------------------
    CONFIG = load_config(args.config)
    for k, v in vars(args).items():
        setattr(CONFIG, k, v)
    log_config(CONFIG, logger=logger)

    # Set seed ----------------------------------
    torch.manual_seed(CONFIG.training.seed)

    # Set up model ------------------------------
    logger.info("Loading model...")
    model, optimizer, scheduler, criterion = set_up_model(CONFIG)
    logger.info("  L Done.")

    # # Set up metrics ----------------------------
    # logger.info("Setting up metrics...")
    # metrics = {
    #     "train": {
    #         "loss": [],
    #         "accuracy": [],
    #     },
    #     "valid": {
    #         "loss": [],
    #         "accuracy": [],
    #     },
    # }
    # logger.info("  L Done.")

    # # Implement the K-fold cross-validation -----
    # logger.info("Implementing K-fold cross-validation...")
    # for fold in range(CONFIG.training.kfold):

    #     logger.info(f"  L Fold {fold + 1}/{CONFIG.training.kfold}")

    #     # Set up dataloaders --------------------
    #     logger.info("  L Setting up dataloaders...")
    #     # train_loader = get_dataloader(
    #     #     CONFIG.data.train,
    #     #     CONFIG.training.batch_size,
    #     #     CONFIG.tokenizer,
    #     #     shuffle=True,
    #     #     drop_last=True,
    #     # )
    #     # valid_loader = get_dataloader(
    #     #     CONFIG.data.valid,
    #     #     CONFIG.training.batch_size,
    #     #     CONFIG.tokenizer,
    #     #     shuffle=False,
    #     #     drop_last=False,
    #     # )
    #     logger.info("  L Done.")

    #     # Train model ---------------------------
    #     logger.info("  L Training model...")
    #     for epoch in range(CONFIG.training.epochs):
    #         logger.info(f"    L Epoch {epoch + 1}/{CONFIG.training.epochs}")

    #         # Train step
    #         logger.info("      L Training step...")
    #         model.train()
    #         for i, (src, tgt) in enumerate(train_loader):
    #             src = src.to(CONFIG.training.device)
    #             tgt = tgt.to(CONFIG.training.device)

    #             optimizer.zero_grad()
    #             output = model(src, tgt)
    #             loss = criterion(output, tgt)
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()

    #             metrics["train"]["loss"].append(loss.item())
    #             logger.info(f"        L Step {i + 1}/{len(train_loader)}: loss={loss.item()}")

    #         # Validation step
    #         logger.info("      L Validation step...")
    #         model.eval()
    #         with torch.no_grad():
    #             for i, (src, tgt) in enumerate(valid_loader):
    #                 src = src.to(CONFIG.training.device)
    #                 tgt = tgt.to(CONFIG.training.device)

    #                 output = model(src, tgt)
    #                 loss = criterion(output, tgt)

    #                 metrics["valid"]["loss"].append(loss.item())
    #                 logger.info(f"        L Step {i + 1}/{len(valid_loader)}: loss={loss.item()}")

    #     logger.info("  L Done.")

    # Set up dataloaders ------------------------
    logger.info("Setting up dataloaders...")
    # train_loader = get_dataloader(
    #     CONFIG.data.train,
    #     CONFIG.training.batch_size,
    #     CONFIG.tokenizer,
    #     shuffle=True,
    #     drop_last=True,
    # )
