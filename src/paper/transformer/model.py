#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import List

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
    ):
        super(Transformer, self).__init__()

        # Information
        self.model_type = "Transformer"
        self.d_model = d_model

        # Layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_token_embedding = TokenEmbedding(num_tokens=src_vocab_size, d_model=d_model)
        self.tgt_token_embedding = TokenEmbedding(num_tokens=tgt_vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=8000
        )  # noqa

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
        src_embedding = self.positional_encoding(
            self.src_token_embedding(src)
        )  # 3D tensor (batch_size, src_len, d_model)  # noqa
        tgt_embedding = self.positional_encoding(
            self.tgt_token_embedding(tgt)
        )  # 3D tensor (batch_size, tgt_len, d_model)  # noqa

        # Transformer
        outs = self.transformer(
            src=src_embedding,
            tgt=tgt_embedding,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)  # 2D tensor (d_model, tgt_vocab_size)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.transformer.encoder(
            self.positional_encoding(self.src_token_embedding(src)),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_token_embedding(tgt)),
            memory=memory,
            tgt_mask=tgt_mask,
        )


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to token embeddings.

    This class generates sinusoidal positional encodings, which are added to the token embeddings
    in a Transformer model to provide information about the position of each token in a sequence.

    Parameters
    ----------
    d_model : int
        The dimension of the token embeddings (embedding size).
    dropout : float, optional
        Dropout probability applied after adding positional encodings (default is 0.1).
    max_len : int, optional
        The maximum length of sequences that the positional encoding will support (default is 5000).

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer applied to the embeddings after positional encodings are added.
    pos_embedding : torch.Tensor
        Precomputed positional encodings stored as a buffer, with shape (1, max_len, d_model).
    """

    def __init__(
        self,
        d_model: int,
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
        d_model : int
            The dimension of the token embeddings.
        dropout : float, optional
            Dropout probability applied after adding positional encodings (default is 0.1).
        max_len : int, optional
            The maximum length of sequences the positional encoding will support (default is 5000).
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encoding
        pos_embedding = torch.zeros(max_len, d_model)  # 2D tensor (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # 2D tensor (max_len, 1)  # noqa
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # 1D tensor (d_model / 2) # noqa
        pos_embedding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pos_embedding[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register the buffer
        pos_embedding = pos_embedding.unsqueeze(0)  # 3D tensor (max_len, 1, d_model)
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
            3D tensor (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            The input tensor with positional encodings added, with the same shape as `x`.
        """
        x = x + self.pos_embedding[:, : x.size(1), :]  # Add positional encodings
        return self.dropout(x)  # 3D tensor (seq_len, batch_size, d_model)


class TokenEmbedding(nn.Module):
    """Token embedding module.

    This class converts token indices into fixed-dimensional embedding vectors,
    and applies scaling by the square root of the embedding dimension to stabilize
    training in Transformer models.

    Parameters
    ----------
    num_tokens : int
        The total number of unique tokens in the vocabulary (vocabulary size).
    d_model : int
        The embedding dimension for each token.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer that converts token indices into embedding vectors.
    d_model : int
        Embedding dimension, used for scaling.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
    ):
        """
        Initializes the token embedding layer and stores the embedding dimension.

        Parameters
        ----------
        num_tokens : int
            The total number of tokens in the vocabulary.
        d_model : int
            The dimension of the embeddings generated for each token.
        """
        super(TokenEmbedding, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_tokens, d_model)  # 2D tensor (num_tokens, d_model)
        self.d_model = d_model

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
            3D tensor (batch_size, seq_len, d_model).
        """
        return self.embedding(tokens) * math.sqrt(self.d_model)


def generate_pad_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Create a padding mask for a tensor.

    This function creates a mask tensor that is `True` where the input tensor is equal to the
    padding index (`PAD_IDX`), and `False` otherwise. This mask can be used to ignore padded
    positions in the input tensor during training.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (batch_size, seq_len) containing token indices.

    Returns
    -------
    torch.Tensor
        Mask tensor of shape (batch_size, seq_len) with `True` values where the input tensor is
        equal to `PAD_IDX`.
    """
    return tensor == PAD_IDX


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """Generate a square subsequent mask for a sequence.

    This function generates a square mask for a sequence, where the mask is a lower triangular
    matrix with values below the diagonal set to `True` and values on or above the diagonal set
    to `False`.

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


def create_masks(src: torch.Tensor, tgt: torch.Tensor) -> tuple:
    """
    Creates the necessary masks for the source and target sequences used in a Transformer model.

    This function generates three types of masks:
    1. A causal mask for the target sequence (`tgt_mask`), preventing the model from attending
       to future positions during training.
    2. Padding masks for both the source and target sequences (`src_padding_mask` and
        `tgt_padding_mask`) to ensure that padded positions (where `PAD_IDX` is present) are
        ignored in the attention mechanisms.

    Parameters
    ----------
    src : torch.Tensor
        The source sequence tensor of shape (batch_size, src_len), containing token indices.
    tgt : torch.Tensor
        The target sequence tensor of shape (batch_size, tgt_len), containing token indices.

    Returns
    -------
    tuple
        A tuple containing three tensors:
        - tgt_mask : torch.Tensor
            A square subsequent mask of shape (tgt_len, tgt_len), where positions above the diagonal
            are masked (`-inf`) to prevent attending to future tokens.
        - src_padding_mask : torch.Tensor
            A mask of shape (batch_size, src_len), where `True` values indicate padded positions.
        - tgt_padding_mask : torch.Tensor
            A mask of shape (batch_size, tgt_len), where `True` values indicate padded positions.
    """
    tgt_mask = generate_square_subsequent_mask(tgt.size(1))  # 2D tensor (tgt_len, tgt_len)

    src_padding_mask = generate_pad_mask(src)  # 2D tensor (batch_size, src_len)
    tgt_padding_mask = generate_pad_mask(tgt)  # 2D tensor (batch_size, tgt_len)

    return tgt_mask, src_padding_mask, tgt_padding_mask
