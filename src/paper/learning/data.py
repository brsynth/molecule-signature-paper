import logging
from typing import List, Tuple
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Dataset -----------------------------------------------------------------------------------------
class TSVDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            file_path: str | Path,
            col_indexes: tuple[int, int],
            token_fns: tuple[callable, callable],
            max_rows: int = -1,
            max_lengths: tuple[int, int] = (100, None),
            bos_idx: int = 1,
            eos_idx: int = 2,
    ):
        self.file_path = file_path
        self.col_indexes = {
            "source": col_indexes[0],
            "target": col_indexes[1],
        }
        self.token_fns = {
            "source": token_fns[0],
            "target": token_fns[1],
        }
        self.max_rows = max_rows
        self.max_lengths = {
            "source": max_lengths[0],
            "target": max_lengths[1],
        }
        self.data = self._load_data()

        # Find longuest source sequence
        if self.max_lengths["source"] is None:
            self.max_lengths["source"] = self._find_longuest_sequence("source")
        else:
            self.max_lengths["source"] = max_lengths[0]

        # Find longuest target sequence
        if self.max_lengths["target"] is None:
            self.max_lengths["target"] = self._find_longuest_sequence("target")
        else:
            self.max_lengths["target"] = max_lengths[1]

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def _load_data(self) -> List[Tuple[str, str]]:
        data = []
        with open(self.file_path, 'r') as f:
            next(f)  # Skip header
            for line_idx, line in enumerate(f):
                if (
                    self.max_rows != -1 and  # -1 means no limit
                    line_idx >= self.max_rows
                ):
                    break
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
        try:
            source_tokens = self.token_fns["source"](source)
            target_tokens = self.token_fns["target"](target)
        except Exception:
            logger.warning(f"Error while tokenizing line {idx}, skipped")
            return None

        # Adding BOS and EOS tokens
        source_tokens = self._add_beos_indexes(source_tokens)
        target_tokens = self._add_beos_indexes(target_tokens)

        # Trimming
        source_tokens = self._trim_sequence(source_tokens, self.max_lengths["source"])
        target_tokens = self._trim_sequence(target_tokens, self.max_lengths["target"])

        return torch.tensor(source_tokens), torch.tensor(target_tokens)

    def __len__(self):
        return len(self.data)


class ListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[str],
        token_fn: callable,
        max_rows: int = -1,
        max_length: int = 100,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        self.data = data
        self.token_fn = token_fn
        self.max_rows = max_rows
        self.max_length = max_length
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.data[idx]

        # Tokenization
        try:
            tokens = self.token_fn(sequence)
        except Exception:
            logger.warning(f"Error while tokenizing line {idx}, skipped")
            return None

        # Adding BOS and EOS tokens
        tokens = [self.bos_idx] + tokens + [self.eos_idx]

        # Trimming
        tokens = self._trim_sequence(tokens, self.max_length)

        return torch.tensor(tokens)

    def _trim_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        if max_length is not None:
            return tokens[:max_length]

        return tokens

    def __len__(self):
        return len(self.data)


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


def collate_fn_simple(batch):
    # Skip eventual None
    batch = [b for b in batch if b is not None]

    # Padding
    src = pad_sequence(batch, batch_first=True, padding_value=3)

    # Masking
    src_mask = torch.zeros(src.size(1), src.size(1)).type(torch.bool)
    src_padding_mask = (src == 3)  # 2D tensor (batch_size, src_len)

    # Return
    return src, None, None, src_mask, None, src_padding_mask, None


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
    # Skip eventual None values
    batch = [d for d in batch if d is not None]

    # Unzip batch
    src, tgt = zip(*batch)

    # Padding
    src = pad_sequence(src, batch_first=True, padding_value=3)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=3)

    # Shift target to force autoregressive behavior
    tgt_input = tgt[:, :-1]  # Remove last token
    tgt_output = tgt[:, 1:]  # Remove first token

    # Masking
    src_mask = torch.zeros(src.size(1), src.size(1)).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))  # 2D tensor (tgt_len, tgt_len)  # noqa
    src_padding_mask = (src == 3)  # 2D tensor (batch_size, src_len)
    tgt_padding_mask = (tgt_input == 3)  # 2D tensor (batch_size, tgt_len)

    return src, tgt_input, tgt_output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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
        src_mask = torch.zeros(src.size(1), src.size(1)).type(torch.bool)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))  # 2D tensor (tgt_len, tgt_len)  # noqa
        src_padding_mask = (src == pad_idx)  # 2D tensor (batch_size, src_len)
        tgt_padding_mask = (tgt_input == pad_idx)  # 2D tensor (batch_size, tgt_len)

        return src, tgt_input, tgt_output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    return collate_fn
