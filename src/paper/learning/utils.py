import logging
from typing import Optional, List, Tuple
from pathlib import Path

import torch
from torch import Tensor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import sentencepiece as spm

from paper.dataset.utils import (
    pretokenizer_for_smiles,
    pretokenizer_for_signature,
    pretokenizer_for_ecfp,
)


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Tokens ------------------------------------------------------------------------------------------
def _do_nothing(x):
    return x


def _remove_spaces(x):
    return x.replace(" ", "")


def _replace_spaces_by_dashes(x):
    return x.replace(" ", "-")


class Tokenizer:

    # Default pretokenizer functions
    default_pre_encode_fn = {
        "SMILES": pretokenizer_for_smiles,
        "SIGNATURE": pretokenizer_for_signature,
        "ECFP": pretokenizer_for_ecfp,
    }

    default_post_decode_fn = {
        "SMILES": _remove_spaces,
        "SIGNATURE": _remove_spaces,
        "ECFP": _replace_spaces_by_dashes,
    }

    def __init__(
        self,
        model_path: str | Path,
        fp_type: Optional[str] = None,
        pre_encode_fn: Optional[callable] = None,
        post_decode_fn: Optional[callable] = None,
    ):
        self.model_path = model_path
        self.model = spm.SentencePieceProcessor(model_file=str(model_path))

        # Set pre-encode function
        if pre_encode_fn is not None:
            self.pre_encode_fn = pre_encode_fn
        elif fp_type is not None:
            try:
                self.pre_encode_fn = self.default_pre_encode_fn[fp_type]
            except KeyError:
                raise ValueError(f"Pre-tokenization function for '{fp_type}' not found")
        else:
            self.pre_encode_fn = _do_nothing

        # Set post-decode function
        if post_decode_fn is not None:
            self.post_decode_fn = post_decode_fn
        elif fp_type is not None:
            try:
                self.post_decode_fn = self.default_post_decode_fn[fp_type]
            except KeyError:
                raise ValueError(f"Post decode function for '{fp_type}' not found")
        else:
            self.post_decode_fn = _do_nothing

    def encode(self, text: str) -> List[str]:
        try:
            text = self.pre_encode_fn(text)
            return self.model.encode_as_ids(text)
        except Exception:
            raise

    # Decode a list of integers or a Tensor to a string
    def decode(self, ids: List[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        text = self.model.decode_ids(ids)
        return self.post_decode_fn(text)

    def __len__(self) -> int:
        return self.model.vocab_size()

    def __call__(self, *args, **kwds):
        """Dispatch method."""
        if isinstance(args[0], str):
            return self.encode(*args, **kwds)

        elif isinstance(args[0], list):
            return self.decode(*args, **kwds)

        elif isinstance(args[0], int):
            return self.decode(*args, **kwds)

        elif isinstance(args[0], Tensor):
            return self.decode(*args, **kwds)

        else:
            raise ValueError("Input must be a string, an integer, or a list of integers")


# Early stopping ----------------------------------------------------------------------------------
class EarlyStoppingCustom(EarlyStopping):
    """Custom early stopping callback for Lightning.

    This callback is a subclass of `pytorch_lightning.callbacks.EarlyStopping` with the following
    modifications:
    - The `threshold_mode` parameter is added to allow for the use of relative thresholds.
    - The `_evaluate_stopping_criteria` method is modified to allow for the use of relative thresholds.
    - The `_improvement_message` method is modified to print more float digits.
    """
    def __init__(self, *args, threshold_mode: str = "rel", **kwargs):
        self.threshold_mode = threshold_mode
        super().__init__(*args, **kwargs)

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.threshold_mode == "abs" and self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        elif self.threshold_mode == 'rel' and self.monitor_op(current, (self.best_score * (1 + self.min_delta * torch.sign(self.best_score))).to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.4f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            if self.threshold_mode == "abs":
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current):.4f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.4f}"
                )
            else:  # self.threshold_mode == "rel":
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current) / abs(self.best_score):.4f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.4f}"
                )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.4f}"
        return msg
