"""Transformer model for sequence-to-sequence tasks."""
import math
import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchmetrics.aggregation import MeanMetric

from paper.learning.data import TSVDataset, collate_fn, generate_square_subsequent_mask

# DEBUG LOGGING -----------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# MODEL ------------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Positional encoding module for adding positional information to token embeddings.

    This class generates sinusoidal positional encodings, which are added to the token embeddings
    in a Transformer model to provide information about the position of each token in a sequence.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer applied to embeddings after positional encodings are added.
    pos_embedding : torch.Tensor
        Precomputed positional encodings stored as a buffer, with shape (1, max_len, dim_model).
    """
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """
        Initializes the PositionalEncoding module.

        Precomputes the sinusoidal positional encodings up to `max_len` positions
        and stores them in a buffer. These encodings are then added to the token
        embeddings during the forward pass.

        Parameters
        ----------
        dim_model : int
            Dimension of the token embeddings (embedding size).
        dropout : float, optional
            Dropout probability applied after adding positional encodings (default is 0.1).
        max_len : int, optional
            Maximum length of sequences that positional encoding will support (default is 5000).
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encoding
        pos_embedding = torch.zeros(max_len, dim_model)  # 2D tensor (max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # 2D tensor (max_len, 1)  # noqa
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )  # 1D tensor (dim_model / 2) # noqa
        pos_embedding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pos_embedding[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register the buffer
        pos_embedding = pos_embedding.unsqueeze(0)  # 3D tensor (1, max_len, dim_model)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input token embeddings.

        The positional encodings are added element-wise to the token embeddings
        to provide information about the position of each token in the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of token embeddings.
            3D tensor (batch_size, seq_len, dim_model)

        Returns
        -------
        torch.Tensor
            The input tensor with positional encodings added, with the same shape as `x`.
        """
        x = x + self.pos_embedding[:, : x.size(1), :]  # Add positional encodings
        return self.dropout(x)  # 3D tensor (batch_size, seq_len, dim_model)


class TokenEmbedding(nn.Module):
    """Token embedding module.

    This class converts token indices into fixed-dimensional embedding vectors,
    and applies scaling by the square root of the embedding dimension to stabilize
    training in Transformer models.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer that converts token indices into embedding vectors.
    dim_model : int
        Embedding dimension, used for scaling.
    """

    def __init__(
        self,
        num_tokens: int,
        dim_model: int,
    ):
        """
        Initializes the token embedding layer and stores the embedding dimension.

        Parameters
        ----------
        num_tokens : int
            Total number of tokens in the vocabulary.
        dim_model : int
            Dimension of the embeddings generated for each token.
        """
        super(TokenEmbedding, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_tokens, dim_model)  # 2D tensor (num_tokens, dim_model)
        self.dim_model = dim_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Applies the embedding layer to the input tokens and scales the embeddings.

        Parameters
        ----------
        tokens : torch.Tensor
            Tensor of token indicescontaining token indices from the vocabulary.
            2D tensor (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Tensor of scaled embedding vectors.
            3D tensor (batch_size, seq_len, dim_model).
        """
        return self.embedding(tokens) * math.sqrt(self.dim_model)


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        # Data arguments
        data_train_file: str | Path,
        data_valid_file: str | Path,
        data_col_indexes: Tuple[int, int],
        data_max_rows: int,  # Useful for debugging
        # Dataloader arguments
        dataloader_num_workers: int,
        dataloader_seq_max_lengths: Tuple[int, int],
        # Tokens arguments
        token_fns: Tuple[Any, Any],
        token_unk_idx: int = 0,
        token_bos_idx: int = 1,
        token_eos_idx: int = 2,
        token_pad_idx: int = 3,
        # Model arguments
        model_dim: int = 512,
        model_num_heads: int = 8,
        model_num_encoder_layers: int = 3,
        model_num_decoder_layers: int = 3,
        model_dropout: float = 0.1,
        model_vocab_sizes: Optional[Tuple[int, int]] = None,
        # Training arguments
        train_batch_size: int = 32,
        train_max_epochs: int = 20,
        # Scheduler arguments
        scheduler_method: str = "plateau",
        scheduler_lr: float = 0.0001,
        scheduler_exponential_gamma: float = 0.95,
        scheduler_plateau_patience: int = 5,
        scheduler_plateau_factor: float = 0.1,
        # Fine-tuning arguments
        finetune_freeze_encoder: bool = False,
        # Distributed related arguments
        ddp_num_nodes: int = 1,
        ddp_num_comp_units: int = 1,
    ) -> None:
        super(TransformerModel, self).__init__()

        # Save all passed arguments
        self.save_hyperparameters()  # hyperparameters saved into self.hparams

        # Data arguments
        self.data_train_file = data_train_file
        self.data_valid_file = data_valid_file
        self.data_col_indexes = data_col_indexes
        self.data_max_rows = data_max_rows

        # Token models
        self.token_fns = token_fns
        self.token_unk_idx = token_unk_idx
        self.token_bos_idx = token_bos_idx
        self.token_eos_idx = token_eos_idx
        self.token_pad_idx = token_pad_idx

        # Dataloader arguments
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_seq_max_lengths = dataloader_seq_max_lengths

        # Model arguments
        self.model_dim = model_dim
        self.model_num_heads = model_num_heads
        self.model_num_encoder_layers = model_num_encoder_layers
        self.model_num_decoder_layers = model_num_decoder_layers
        self.model_dropout = model_dropout
        self.model_vocab_sizes = self._get_vocabulary_sizes() if model_vocab_sizes is None else model_vocab_sizes  # noqa

        # Model components
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=model_num_heads,
            num_encoder_layers=model_num_encoder_layers,
            num_decoder_layers=model_num_decoder_layers,
            dim_feedforward=model_dim * 4,
            dropout=model_dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(model_dim, self.model_vocab_sizes[1])
        self.src_token_embedding = TokenEmbedding(num_tokens=self.model_vocab_sizes[0], dim_model=model_dim)  # noqa
        self.tgt_token_embedding = TokenEmbedding(num_tokens=self.model_vocab_sizes[1], dim_model=model_dim)  # noqa
        self.positional_encoding = PositionalEncoding(dim_model=model_dim, dropout=model_dropout, max_len=8000)  # noqa

        # Initialize weights
        self.init_weights()

        # Training arguments
        self.train_batch_size = train_batch_size
        self.train_max_epochs = train_max_epochs

        # Scheduler arguments
        self.scheduler_method = scheduler_method
        self.scheduler_lr = scheduler_lr
        self.scheduler_exponential_gamma = scheduler_exponential_gamma
        self.scheduler_plateau_patience = scheduler_plateau_patience
        self.scheduler_plateau_factor = scheduler_plateau_factor

        # Fine-tuning arguments
        self.finetune_freeze_encoder = finetune_freeze_encoder
        if self.finetune_freeze_encoder:
            self.freeze_encoder()

        # Distributed-computing related arguments
        self.ddp_num_nodes = ddp_num_nodes
        self.ddp_num_comp_units = ddp_num_comp_units

        # Loss function
        self.loss = nn.CrossEntropyLoss()

        # Aggregators
        self.train_loss = MeanMetric()
        self.train_acc = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Embedding + Position Encoding
        src_embedding = self.positional_encoding(self.src_token_embedding(src))  # 3D tensor (batch_size, src_len, dim_model)  # noqa
        tgt_embedding = self.positional_encoding(self.tgt_token_embedding(tgt))  # 3D tensor (batch_size, tgt_len, dim_model)  # noqa

        # Model forward pass
        logits = self.transformer(
            src=src_embedding,
            tgt=tgt_embedding,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(logits)  # 2D tensor (dim_model, tgt_vocab_size)

    def training_step(self, batch, batch_idx: int):
        # Compute loss and accuracy
        values = self._shared_step(batch, batch_idx)

        # Accumulate for epoch end
        self.train_loss.update(values["loss"])
        self.train_acc.update(values["accuracy"])

        # Quick log for progress bar
        self.log("train/loss", values["loss"], on_step=True, on_epoch=False, prog_bar=True)  # noqa

        return values["loss"]  # Return loss for backward pass

    def on_train_epoch_end(self) -> None:
        # Compute epoch loss and accuracy (MeanMetric objects)
        epoch_loss = self.train_loss.compute()
        epoch_acc = self.train_acc.compute()

        # Get the learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Log (x-axis is epochs)
        self.logger.experiment.add_scalar("train/epoch/loss", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar("train/epoch/acc", epoch_acc, self.current_epoch)
        self.logger.experiment.add_scalar("train/epoch/lr", lr, self.current_epoch)  # noqa

        # Log through Lightning log (x-axis is steps)
        # self.logger.experiment.add_scalar("train/epoch/loss", epoch_loss, self.current_epoch)  # noqa
        # self.logger.experiment.add_scalar("train/epoch/acc", epoch_acc, self.current_epoch)  # noqa

        # Reset the aggregators (not done if there's no call to self.log or self.log_dict)
        self.train_loss.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx: int) -> None:
        # Compute loss and accuracy
        values = self._shared_step(batch, batch_idx)

        # Aggregate for epoch end
        self.val_loss.update(values["loss"])
        self.val_acc.update(values["accuracy"])

    def on_validation_epoch_end(self):
        # Compute epoch loss and accuracy (MeanMetric objects)
        epoch_loss = self.val_loss.compute()
        epoch_acc = self.val_acc.compute()

        # Log (x-axis is epochs)
        self.logger.experiment.add_scalar("val/epoch/loss", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar("val/epoch/acc", epoch_acc, self.current_epoch)

        # Log through Lightning log (x-axis is steps)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True, reduce_fx="mean")
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, reduce_fx="mean")

    def configure_optimizers(self):
        # Optimizer: AdamW ---------------------------------------------------
        # Get trainable parameters
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))

        # Split parameters into those that should be decayed and those that should not
        decay_params = []
        no_decay_params = []
        for name, param in trainable_params:
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.scheduler_lr,
        )

        # Scheduler : Cyclic --------------------------------------------------
        if self.scheduler_method == "cyclic":
            steps_per_epoch = len(self.train_dataloader())
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.scheduler_lr / 100,
                    max_lr=self.scheduler_lr,
                    step_size_up=steps_per_epoch * 5,  # 5 epochs
                    cycle_momentum=False,
                ),
                "interval": "step",
                "frequency": 1,
            }

        # Scheduler: One Cycle ------------------------------------------------
        elif self.scheduler_method == "one-cycle":
            steps_per_epoch = len(self.train_dataloader())
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.scheduler_lr,
                    total_steps=self.train_max_epochs * steps_per_epoch,
                    pct_start=0.05,
                    anneal_strategy="cos",
                    div_factor=10,
                    final_div_factor=1,
                ),
                "interval": "step",
                "frequency": 1,
            }

        # Scheduler: Cosine Annealing with Warm Restarts ----------------------
        elif self.scheduler_method == "cosine-warm-restarts":
            total_steps = len(self.train_dataloader()) * self.train_max_epochs
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=total_steps // 5,  # 5% of total steps before the first restart
                    T_mult=1,  # Multiplicative factor by which T_i is extended after each restart
                    eta_min=self.scheduler_lr / 100,
                ),
                "interval": "step",
                "frequency": 1,
            }

        # Scheduler: Exponential ----------------------------------------------
        elif self.scheduler_method == "exponential":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=self.scheduler_exponential_gamma,
                ),
                "interval": "epoch",
                "frequency": 1,
            }

        # Scheduler: ReduceLROnPlateau ----------------------------------------
        elif self.scheduler_method == "plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_plateau_factor,
                    patience=self.scheduler_plateau_patience,
                    # verbose=True,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        # Fallback is to raise an error
        else:
            raise ValueError(f"Unknown scheduler method: {self.scheduler_method}")

        return [optimizer], [scheduler]

    def _shared_step(self, batch, batch_idx) -> dict:
        source, target_input, target_output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = batch  # noqa
        logits = self(
            source,
            target_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        # Reshape logits and target_output if necessary
        device = logits.device
        if device.type == "cpu":
            logits = logits.contiguous()
            target_output = target_output.contiguous()

        # Loss
        loss = self.loss(
            logits.reshape(-1, logits.size(-1)),  # y_hat
            target_output.reshape(-1)  # y
        )

        # Accuracy
        target_mask = (target_output != self.token_pad_idx)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_target = target_output.view(-1)
        flat_mask = target_mask.view(-1)
        valid_logits = flat_logits[flat_mask]  # Remove padding tokens
        valid_target = flat_target[flat_mask]  # Remove padding tokens
        acc = accuracy(
            valid_logits,
            valid_target,
            task="multiclass",
            num_classes=self.model_vocab_sizes[1]
        )
        # Metrics
        return {
            "loss": loss,
            "accuracy": acc,
        }

    def prepare_data(self) -> None:
        # Unzip data file if necessary
        msg = (
            "Unzipping data files is not yet supported. Please consider "
            "unzipping the data train and valid files manually."
        )
        if self.data_train_file.suffix == ".gz" or self.data_valid_file.suffix == ".gz":
            raise NotImplementedError(msg)

    def setup(self, stage: str):

        if stage == "fit" or stage is None:
            # Load datasets
            self.train_dataset = TSVDataset(
                file_path=self.data_train_file,
                col_indexes=self.data_col_indexes,
                max_rows=self.data_max_rows,
                token_fns=(self.token_fns[0], self.token_fns[1]),
                max_lengths=self.dataloader_seq_max_lengths,
            )
            self.valid_dataset = TSVDataset(
                file_path=self.data_valid_file,
                col_indexes=self.data_col_indexes,
                token_fns=(self.token_fns[0], self.token_fns[1]),
                max_rows=self.data_max_rows,
                max_lengths=self.dataloader_seq_max_lengths,
            )

    def train_dataloader(self):
        num_workers = self.dataloader_num_workers
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.dataloader_num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        num_workers = self.dataloader_num_workers
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.dataloader_num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    def set_decoding_strategy(
        self,
        strategy: str = "greedy",
        max_length: int = 64,
        beam_size: int = 5,
        **kwargs
    ) -> None:
        """Set the decoding strategy and additional keyword arguments for prediction.

        Parameters
        ----------
        strategy : str
            The decoding strategy to use (e.g., "greedy" or "beam").
        **kwargs : dict
            Additional keyword arguments for the prediction strategy.
        """
        self.decoding_strategy = strategy
        self.max_length = max_length
        self.beam_size = beam_size
        self.decoding_kwargs = kwargs

    def predict_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        strategy = getattr(self, 'decoding_strategy', 'greedy')
        if strategy == 'greedy':
            return self._greedy_search_batch(batch)
        elif strategy == 'beam':
            return self._beam_search_batch(batch)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

    def _greedy_search_batch(
        self,
        batch: List[torch.Tensor]
    ) -> List[List[Tuple[torch.Tensor, float]]]:
        """Generate predictions using greedy decoding.

        Parameters
        ----------
        batch : List[torch.Tensor]
            A batch of source sequences to generate target sequences for.

        Returns
        -------
        List[List[Tuple[torch.Tensor, float]]]
            Generated target sequences and their scores.
        """
        # Get decoding settings
        err = "Decoding settings not configured. Call `set_decoding_strategy` first."
        try:
            max_length = self.max_length
        except AttributeError:
            raise ValueError(err)

        # Unpack the batch
        source, _, _, src_mask, _, src_padding_mask, _ = batch
        batch_size = source.size(0)

        # Encode source
        src_embedding = self.positional_encoding(self.src_token_embedding(source))  # (batch_size, src_len, dim_model)  # noqa
        memory = self.transformer.encoder(
            src_embedding, mask=src_mask, src_key_padding_mask=src_padding_mask
        )  # (batch_size, src_len, dim_model)

        # Initialize targets with BOS token
        targets = torch.full(
            size=(batch_size, 1),
            fill_value=self.token_bos_idx,
            dtype=torch.long,
            device=self.device
        )  # (batch_size, 1)

        # Init storage for scores
        target_scores = torch.zeros(batch_size, device=self.device)  # (batch_size,)

        # Track if sequences have reached EOS token
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)  # (batch_size,)

        # Autoregressive decoding
        for _ in range(1, max_length):
            # Generate target mask to prevent full peeking
            tgt_mask = generate_square_subsequent_mask(targets.size(1))

            # Embed target
            tgt_embedding = self.positional_encoding(self.tgt_token_embedding(targets))  # (batch_size, tgt_len, dim_model)  # noqa

            # Decode target
            output = self.transformer.decoder(
                tgt=tgt_embedding,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,  # No padding in target
                memory_key_padding_mask=src_padding_mask,  # No padding in memory
            )

            # Generate logits
            logits = self.generator(output[:, -1, :])  # (batch_size, tgt_vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Get the most likely token (greedy decoding)
            next_token_log_prob, next_token = torch.max(log_probs, dim=-1)  # (batch_size,)

            # Update scores (for unfinished sequences)
            target_scores += next_token_log_prob * (~is_finished).float()

            # Append the predicted token to the target sequence
            next_token = next_token.unsqueeze(-1)  # (batch_size, 1)
            targets = torch.cat([targets, next_token], dim=1)  # (batch_size, seq_len + 1)

            # Update finished sequences
            is_finished |= next_token.squeeze(-1) == self.token_eos_idx

            # Break if all sequences have an EOS token
            if torch.all(is_finished):
                break

        # Return generated sequences along with scores
        results = []
        for i in range(batch_size):
            seq = targets[i]
            score = target_scores[i].item()
            results.append((seq, score))

        # Append an extra dimension for compatibility with beam search
        for i in range(batch_size):
            results[i] = [results[i]]  # (1, tgt_len)

        return results  # (batch_size, 1, tgt_len)

    def _beam_search_batch(
        self,
        batch: List[torch.Tensor],
    ) -> List[List[Tuple[torch.Tensor, float]]]:
        """Beam search batched, leveraging GPU and deduplication for each iteration.

        The method returns for each source sequence in the batch a top-k (beam_size)
        of generated sequences, along with their scores (sum of log_probs).

        Parameters
        ----------
        batch : List[torch.Tensor]
            A batch of tensors containing at least:
            - source : torch.Tensor (batch_size, src_len)
            - src_mask : torch.Tensor or None (optional, ex: (src_len, src_len))
            - src_padding_mask : torch.Tensor or None (optional, ex: (batch_size, src_len))
            Other elements in the batch are not used here.

        Returns
        ------
        all_generated_sequences : List[List[Tuple[torch.Tensor, float]]]
            For each source sequence in the batch, a list of (sequence, score).
        """
        # -- 1) Get decoding settings and tensors from the batch
        source, _, _, src_mask, _, src_padding_mask, _ = batch
        batch_size = source.size(0)

        msg = "Decoding settings not configured. Call `set_decoding_strategy` first."
        try:
            max_length = self.max_length
            beam_size = self.beam_size
        except AttributeError:
            raise ValueError(msg)

        # -- 2) Encode source sequences in the batch
        # Shape : (batch_size, src_len, dim_model)
        src_embedding = self.positional_encoding(self.src_token_embedding(source))
        # memory : (batch_size, src_len, dim_model)
        memory = self.transformer.encoder(
            src_embedding,
            mask=src_mask,  # ex: (src_len, src_len)
            src_key_padding_mask=src_padding_mask,  # ex: (batch_size, src_len)
        )

        # -- 3) Repeat memory for beam_size
        # Each element of the batch will be repeated beam_size times
        # => (batch_size * beam_size, src_len, dim_model)
        expanded_memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1)
        expanded_memory = expanded_memory.view(batch_size * beam_size, memory.size(1), memory.size(2))  # noqa

        # Same for the source padding mask
        # => (batch_size * beam_size, src_len)
        if src_padding_mask is not None:
            expanded_src_padding_mask = src_padding_mask.unsqueeze(1).repeat(1, beam_size, 1)
            expanded_src_padding_mask = expanded_src_padding_mask.view(
                batch_size * beam_size, src_padding_mask.size(1)
            )
        else:
            expanded_src_padding_mask = None

        # -- 4) Init output sequences and scores
        # All sequences will start with BOS token
        # sequences : (batch_size * beam_size, 1) with token_bos_idx
        device = self.device
        bos_tokens = torch.full(
            (batch_size * beam_size, 1),
            fill_value=self.token_bos_idx,
            dtype=torch.long,
            device=device,
        )
        sequences = bos_tokens  # will contain generated tokens
        # scores : (batch_size * beam_size,)
        # scores starts at 0.0 for each sequence (sum of log_probs)
        scores = torch.zeros(batch_size * beam_size, device=device)

        # Track execution time needed for each sequence
        timers = torch.zeros(batch_size * beam_size, device=device)

        # Track if sequences are terminated (reached EOS token)
        # False = still active ; True = terminated
        is_finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

        # Precompute look-ahead mask 2D : (max_length, max_length)
        # will be sliced at each step
        full_look_ahead_mask = generate_square_subsequent_mask(max_length).to(device)

        # Track time
        start_time = time.perf_counter()

        # -- 5) Generation loop
        for step in range(max_length - 1):  # -1 as we already have the BOS token
            # Stop if all sequences are terminated
            if is_finished.all():
                break

            seq_len = sequences.size(1)

            # 5.1) Prepare for decoding
            # Embed target
            tgt_embedding = self.positional_encoding(self.tgt_token_embedding(sequences))
            # Target mask : (seq_len, seq_len)
            tgt_mask = full_look_ahead_mask[:seq_len, :seq_len]

            # 5.2) Go through the decoder
            # output : (batch_size * beam_size, seq_len, dim_model)
            output = self.transformer.decoder(
                tgt=tgt_embedding,
                memory=expanded_memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=expanded_src_padding_mask,
            )
            # We take the last index of each sequence to predict the next token
            # logits : (batch_size * beam_size, vocab_size)
            logits = self.generator(output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size * beam_size, vocab_size)

            # 5.3) For each element of the batch, we get beam_size (or k) expansions
            # Final shape will be (batch_size * beam_size, k) as we keep top k
            # with k = min(beam_size, vocab_size)
            vocab_size = log_probs.size(-1)
            k = min(beam_size, vocab_size)
            topk_log_probs, topk_indices = log_probs.topk(k, dim=-1)  # (batch_size * beam_size, k)

            # 5.4) Construction of all new candidates sequences
            # We build (batch_size * beam_size * k) candidates before filtering by batch
            # new_candidates : (batch_index, [ (seq, score), ... ])
            #
            # We will regroup by batch_index (i in [0..batch_size-1]) then filter to keep only
            # the top beam_size sequences
            #
            # For readability, we will reorganize the tensors in (batch, beam_size) shape
            # shape => (batch_size, beam_size, k)
            topk_log_probs = topk_log_probs.view(batch_size, beam_size, k)
            topk_indices = topk_indices.view(batch_size, beam_size, k)
            old_scores = scores.view(batch_size, beam_size)
            old_timers = timers.view(batch_size, beam_size)
            old_sequences = sequences.view(batch_size, beam_size, -1)
            old_is_finished = is_finished.view(batch_size, beam_size)

            new_beam_sequences = []
            new_beam_scores = []
            new_beam_timers = []

            # Register computing time
            exec_time = time.perf_counter() - start_time

            # Go through each element of the batch
            for i in range(batch_size):
                candidates_dict = {}  # key: tuple(seq), value: (seq_tensor, score, timer)

                for b in range(beam_size):

                    if old_is_finished[i, b]:

                        # If the sequence is already finished, we copy it as is
                        seq_finished = old_sequences[i, b]
                        sc_finished = old_scores[i, b].item()
                        tm_finished = old_timers[i, b].item()

                        # plus, we force the addition of a padding token to keep the shape
                        dummy_token = torch.tensor([self.token_pad_idx], device=device)
                        new_seq = torch.cat([seq_finished, dummy_token], dim=0)

                        key = tuple(new_seq.tolist())
                        # Prevent duplication
                        if key not in candidates_dict:
                            # Store sequence, score, and computing time
                            candidates_dict[key] = (new_seq, sc_finished, tm_finished)
                        else:
                            # If already present, we keep the best score and update execution time
                            if sc_finished > candidates_dict[key][1]:
                                candidates_dict[key] = (new_seq, sc_finished, tm_finished)
                        continue

                    # Else (not finished), we get k new expansions
                    base_seq = old_sequences[i, b]
                    base_score = old_scores[i, b].item()

                    for j in range(k):
                        next_token = topk_indices[i, b, j].unsqueeze(0)  # shape (1,)
                        next_score = base_score + topk_log_probs[i, b, j].item()

                        new_seq = torch.cat([base_seq, next_token], dim=0)
                        key = tuple(new_seq.tolist())
                        if key not in candidates_dict:
                            candidates_dict[key] = (new_seq, next_score, exec_time)
                        else:
                            # If already present, we keep the best score
                            if next_score > candidates_dict[key][1]:
                                candidates_dict[key] = (new_seq, next_score, exec_time)

                # From there, we have all the candidates for the batch element i. We
                # will filter and keep only the top beam_size sequences (as we may have more)
                # Filtering is done by score
                all_candidates = list(candidates_dict.values())  # -> List[(seq, score, exec_time)]
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidates = all_candidates[:beam_size]

                # If we have less than beam_size candidates, we add dummy sequences to keep
                # the shape (beam_size, seq_len)
                while len(best_candidates) < beam_size:
                    dummy_seq = torch.tensor(
                        [self.token_bos_idx] + [self.token_unk_idx] * (seq_len),
                        device=device
                    )
                    best_candidates.append((dummy_seq, float("-inf"), exec_time))

                # Best candidates are stored in new_beam_sequences, new_beam_scores, new_beam_timers
                # which will be reshaped later
                new_beam_sequences.append([c[0] for c in best_candidates])
                new_beam_scores.append([c[1] for c in best_candidates])
                new_beam_timers.append([c[2] for c in best_candidates])

            # 5.5) Rebuidling the sequences, scores, timers, is_finished tensors
            # - new_beam_sequences[i] is a list of beam_size Tensors for the query i
            # - new_beam_scores[i] is a list of beam_size scores for the query i
            # - new_beam_timers[i] is a list of beam_size execution times for the query i
            # Data are stacked in the next batch
            stacked_sequences = []
            stacked_scores = []
            stacked_timers = []
            stacked_finished = []
            for i in range(batch_size):
                seqs = new_beam_sequences[i]
                scs = new_beam_scores[i]
                timers = new_beam_timers[i]

                # Stacked as 0 => (beam_size, seq_len+1)
                seq_tensor = torch.stack(seqs, dim=0).to(device)
                score_tensor = torch.tensor(scs, device=device, dtype=torch.float)
                timer_tensor = torch.tensor(timers, device=device, dtype=torch.float)

                # Look for terminated sequences
                # Terminated sequences are sequences that end with EOS token or PAD token
                eos_mask = (seq_tensor[:, -1] == self.token_eos_idx) | (seq_tensor[:, -1] == self.token_pad_idx)  # noqa
                stacked_sequences.append(seq_tensor)
                stacked_scores.append(score_tensor)
                stacked_timers.append(timer_tensor)
                stacked_finished.append(eos_mask)

            # Everything is merged in a single tensor of shape (batch_size * beam_size, seq_len+1)
            # Same for scores and is_finished tensors
            sequences = torch.cat(stacked_sequences, dim=0)   # (batch_size * beam_size, seq_len+1)
            scores = torch.cat(stacked_scores, dim=0)         # (batch_size * beam_size,)
            timers = torch.cat(stacked_timers, dim=0)         # (batch_size * beam_size,)
            is_finished = torch.cat(stacked_finished, dim=0)  # (batch_size * beam_size,)

        # -- 6) Regroup terminated sequences for each element of the batch
        # Note: if some sequences are not finished at max_length, we keep them
        # We will sort by score and keep beam_size sequences

        # Reshape sequences and scores to (batch_size, beam_size, seq_len)
        final_sequences = sequences.view(batch_size, beam_size, -1)
        final_scores = scores.view(batch_size, beam_size)
        final_timers = timers.view(batch_size, beam_size)

        all_generated_sequences: List[List[Tuple[torch.Tensor, float, float]]] = []

        for i in range(batch_size):
            seqs_i = final_sequences[i]
            scores_i = final_scores[i]
            timers_i = final_timers[i]

            # Rank sequences by score
            idx_sorted = torch.argsort(scores_i, descending=True)
            sorted_seqs = seqs_i[idx_sorted]
            sorted_scores = scores_i[idx_sorted]
            sorted_timers = timers_i[idx_sorted]

            # Conversion to list[ (seq_tensor, score), ... ]
            sequences_scored = []
            for seq_t, sc, tm in zip(sorted_seqs, sorted_scores, sorted_timers):
                sequences_scored.append((seq_t, float(sc.item()), float(tm.item())))

            # Keep only top beam_size sequences
            best = sequences_scored[:beam_size]
            all_generated_sequences.append(best)

        return all_generated_sequences

    def _get_vocabulary_sizes(self) -> Tuple[int, int]:
        return (
            len(self.token_fns[0]),
            len(self.token_fns[1]),
        )

    def freeze_encoder(self):
        """Freeze the encoder parameters of the transformer model."""
        # Flag for later use
        self.finetune_freeze_encoder = True

        # Freeze the encoder parameters
        for param in self.transformer.encoder.parameters():
            param.requires_grad = False

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable parameters: {trainable_params} / {all_params}")
