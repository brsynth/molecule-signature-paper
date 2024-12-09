from dataclasses import dataclass, field, replace, asdict, is_dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:

    # General option ------------------------------------------------------------------------------
    base_dir: Path = field(default_factory=Path.cwd)
    mode: str = "dummy"  # Choices: translate, prepare, tokenize

    # Folder tree options -------------------------------------------------------------------------
    db: str = "metanetx"  # Choices: metanetx, emolecules, download

    # Optional attributes for "translate" mode ----------------------------------------------------
    source: Optional[str] = None
    target: Optional[str] = None

    # Download options ----------------------------------------------------------------------------
    download_dir: Path = field(init=False)
    download_url: str = field(init=False)
    download_file: Path = field(init=False)

    # Dataset options -----------------------------------------------------------------------------
    dataset_dir: Path = field(init=False)
    dataset_reshaped_file: Path = field(init=False)
    dataset_filtered_file: Path = field(init=False)
    dataset_fingerprints_file: Path = field(init=False)
    dataset_deduped_file: Path = field(init=False)
    dataset_deduped_ext_file: Path = field(init=False)
    dataset_ready_file: Path = field(init=False)

    # Dataset shape -----------------------------------------------------------------------------
    data_col_indexes: dict[str, int] = field(init=False)

    # Sampling options ----------------------------------------------------------------------------
    sample_dir: Path = field(init=False)
    sample_sizes: list[int] = field(default_factory=lambda: [1000, 10000, 5000000])
    sample_files: list[Path] = field(init=False)
    sample_seed: int = 42

    # Splitting options ---------------------------------------------------------------------------
    # Notices
    # -------
    # split_method: str
    #     This is the splitting method to use. Choices are:
    #     - "kfold": K-Fold cross-validation
    #     - "train_valid_split": Train/valid split
    #
    # split_test_proportion: float
    #     This is the proportion of the test set. This value should be between 0 and 1.
    #
    # split_valid_proportion: float
    #     This is the proportion of the validation set. This value should be between 0 and 1.
    #     It is only used when the split_method is "train_valid_split".
    #
    # split_num_folds: int
    #     This is the number of folds to use for K-Fold cross-validation. This value should be
    #     greater than 1. It is only used when the split_method is "kfold".
    #
    split_dir: Path = field(init=False)
    split_from_file: Path = field(init=False)
    split_test_file: Path = field(init=False)
    split_train_files: list[Path] = field(init=False)
    split_valid_files: list[Path] = field(init=False)
    split_method: str = "kfold"  # Choices: kfold or train_valid_split
    split_num_folds: int = 5  # Only for kfold
    split_test_proportion: float = 0.1
    split_valid_proportion: float = 0.1  # Only for train_valid_split
    split_seed: int = 42

    # Analysis options ---------------------------------------------------------------------------
    analysis_dir: Path = field(init=False)

    # Comparative options ------------------------------------------------------------------------
    comparative_dir: Path = field(init=False)

    # Tokenization options ------------------------------------------------------------------------
    # Notices
    # -------
    # token_method:
    #     This is the tokenization method to use. Choices are:
    #     - "bpe": Byte Pair Encoding
    #     - "word": Word tokenization
    #     - "unigram": Unigram tokenization
    #
    # token_vocabulary_coverage: float (not used):
    #     This is the coverage of the vocabulary, i.e. the proportion of the
    #     total number of tokens that should be covered by the vocabulary. Ths
    #     value should be between 0 and 1, and is used to determine the
    #     vocabulary size.
    #
    # token_min_presence: float
    #     This is the minimum frequency a token should appear in fingerprints to
    #     be included in the vocabulary. This value should be between 0 and 1. A
    #     value of 0.01 means that a token should appear in at least 1% of the
    #     fingerprints to be included in the vocabulary.
    #
    token_dir: Path = field(init=False)
    token_from_file: Path = field(init=False)
    token_fingerprints_types: list[str] = field(init=False)
    token_unk_id: int = 0
    token_bos_id: int = 1
    token_eos_id: int = 2
    token_pad_id: int = 3
    token_method: str = "word"  # Choices: bpe, word, unigram
    token_vocabulary_coverage: float = 0.99999  # 1.0 for full coverage
    token_min_presence: float = 0.0001
    token_vocabulary_sizes: dict[str, int] = field(init=False)
    token_models: dict = field(init=False)

    # Train / valid files options -----------------------------------------------------------------
    train_fold_index: int = 0  # Only for kfold method
    data_train_file: Path = field(init=False)
    data_valid_file: Path = field(init=False)

    # Data options -------------------------------------------------------------------------------
    data_max_rows: int = -1

    # Dataloader options --------------------------------------------------------------------------
    dataloader_num_workers: int = 4
    dataloader_seq_max_lengths: dict[str, int] = field(init=False)

    # Model options -------------------------------------------------------------------------------
    model_dim: int = 512
    model_num_heads: int = 8
    model_num_encoder_layers: int = 3
    model_num_decoder_layers: int = 3
    model_dropout: float = 0.1
    # model_vocab_sizes: dict[str, int] = field(init=False)

    # Training options ----------------------------------------------------------------------------
    train_max_epochs: int = 20
    train_batch_size: int = 32
    train_accumulate_grad_batches: int = 1
    train_val_check_interval: float = 1.0
    train_seed: int = 42

    # Fine-tuning options
    finetune: bool = False
    finetune_lr: float = 0.0001
    finetune_freeze_encoder: bool = False

    # Scheduler options
    scheduler_method: str = "plateau"
    scheduler_lr: float = 0.0001
    scheduler_exponential_gamma: float = 0.95
    scheduler_plateau_patience: int = 5
    scheduler_plateau_factor: float = 0.1

    # Early stopping options
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0001

    # Additional training options
    enable_mixed_precision: bool = True

    # Distributed computation options
    ddp_num_nodes: int = 1
    ddp_sync_dist: bool = False
    ddp_num_comp_units: int = 1

    # Output options ------------------------------------------------------------------------------
    out_dir: Path = field(init=False)
    out_sub_dir: Path = None
    out_save_weights_only: bool = False

    # Prediction options --------------------------------------------------------------------------
    pred_test_file: Path = field(init=False)
    pred_max_rows: int = -1
    pred_max_lengths: dict[str, int] = field(init=False)
    pred_max_length: int = field(init=False)
    pred_batch_size: int = 32
    pred_mode: str = "greedy"  # greedy or beam
    pred_beam_size: int = 10

    def __post_init__(self):
        # Download --------------------------------------------------------------------------------
        self.download_dir = self.base_dir / "data" / self.db / "download"
        self.download_url = {
            "metanetx": "https://www.metanetx.org/ftp/4.4/chem_prop.tsv",
            "emolecules": "https://downloads.emolecules.com/free/2024-07-01/version.smi.gz",
        }[self.db]
        self.download_file = {
            "metanetx": self.download_dir / "mnx_raw_4_4.tsv",
            "emolecules": self.download_dir / "emol_raw_2024-07-01.tsv.gz",
        }[self.db]

        # Dataset ---------------------------------------------------------------------------------
        self.dataset_dir = self.base_dir / "data" / self.db / "dataset"
        self.dataset_reshaped_file = self.dataset_dir / "1_reshaped.tsv"
        self.dataset_filtered_file = self.dataset_dir / "2_filtered.tsv"
        self.dataset_fingerprints_file = self.dataset_dir / "3_fingerprints.tsv"
        self.dataset_deduped_file = self.dataset_dir / "4_deduped.tsv"
        self.dataset_deduped_ext_file = self.dataset_dir / "5_deduped_ext.tsv"

        # Dataset ready file ----------------------------------------------------------------------
        if self.db == "metanetx":
            self.dataset_ready_file = self.dataset_deduped_file
        elif self.db == "emolecules":
            self.dataset_ready_file = self.dataset_deduped_ext_file
        else:
            raise ValueError(f"Database '{self.db}' unknown")

        # Dataset shape ---------------------------------------------------------------------------
        self.data_col_indexes = {
            "SMILES": 2,
            "SIGNATURE": 3,
            "ECFP": 4,
        }

        # Sampling --------------------------------------------------------------------------------
        self.sample_dir = self.base_dir / "data" / self.db / "sampling"
        self.sample_files = [
            self.sample_dir / "{i}-sample_{size}.tsv".format(i=i, size=f"{size/1000:.0f}k")
            for i, size in enumerate(self.sample_sizes)
        ]

        # Splitting -------------------------------------------------------------------------------
        self.split_dir = self.base_dir / "data" / self.db / "splitting"
        self.split_from_file = self.sample_files[-1]
        self.split_test_file = self.split_dir / "test.tsv"

        if self.split_method == "kfold":
            self.split_train_files = [
                self.split_dir / f"train_fold{fold}.tsv" for fold in range(self.split_num_folds)
            ]
            self.split_valid_files = [
                self.split_dir / f"valid_fold{fold}.tsv" for fold in range(self.split_num_folds)
            ]

        elif self.split_method == "train_valid_split":
            self.split_train_files = [self.split_dir / "train.tsv"]
            self.split_valid_files = [self.split_dir / "valid.tsv"]

        else:
            raise ValueError(f"Split method '{self.split_method}' unknown")

        # Analysis --------------------------------------------------------------------------------
        self.analysis_dir = self.base_dir / "data" / self.db / "analysis"

        # Comparisons -----------------------------------------------------------------------------
        self.comparative_dir = self.analysis_dir / "comparative"

        # Tokens ----------------------------------------------------------------------------------
        self.token_dir = self.base_dir / "data" / self.db / "tokens"

        # Source file
        if self.db == "metanetx":
            self.token_from_file = self.dataset_ready_file
        elif self.db == "emolecules":
            self.token_from_file = self.sample_files[-1]  # Last (i.e. biggest) sample file
        else:
            self.token_from_file = self.dataset_ready_file

        self.token_fingerprints_types = ["SMILES", "SIGNATURE", "ECFP"]

        # Vocabulary sizes
        if self.token_method == "bpe":
            self.token_vocabulary_sizes = {"SMILES": 128, "SIGNATURE": 512, "ECFP": 2048}

        if self.token_method == "unigram":
            self.token_vocabulary_sizes = {"SMILES": 128, "SIGNATURE": 512, "ECFP": 2048}

        elif self.token_method == "word":
            # 0 for dynamic vocabulary sizes (based on coverage)
            self.token_vocabulary_sizes = {"SMILES": 0, "SIGNATURE": 0, "ECFP": 0}

        else:
            self.token_vocabulary_sizes = {"SMILES": 0, "SIGNATURE": 0, "ECFP": 0}

        # SentencePiece token models
        self.token_models = {
            "SMILES": self.token_dir / "SMILES.model",
            "SIGNATURE": self.token_dir / "SIGNATURE.model",
            "ECFP": self.token_dir / "ECFP.model",
        }

        # Training --------------------------------------------------------------------------------
        # Data files
        if self.split_method == "kfold":
            self.data_train_file = self.split_train_files[self.train_fold_index]
            self.data_valid_file = self.split_valid_files[self.train_fold_index]

        elif self.split_method == "train_valid_split":
            self.data_train_file = self.split_train_files[0]
            self.data_valid_file = self.split_valid_files[0]

        else:
            raise ValueError(f"Split method '{self.split_method}' unknown")

        # Length of sequences
        if self.token_method == "word":
            # Length are set to fully cover fingerprints in 99% of cases (assessed from metanetx)
            self.dataloader_seq_max_lengths = {"SMILES": 64, "SIGNATURE": 611, "ECFP": 98}
        else:
            self.dataloader_seq_max_lengths = {"SMILES": 64, "SIGNATURE": 3072, "ECFP": 512}

        # Lightning logs --------------------------------------------------------------------------
        self.out_dir = self.base_dir / "llogs"

        # Translate handy shorcuts ----------------------------------------------------------------
        if self.source is not None:
            self.source_col_idx = self.data_col_indexes[self.source]
            self.source_max_length = self.dataloader_seq_max_lengths[self.source]
            self.source_token_model = str(self.token_models[self.source])

        if self.target is not None:
            self.target_col_idx = self.data_col_indexes[self.target]
            self.target_max_length = self.dataloader_seq_max_lengths[self.target]
            self.target_token_model = str(self.token_models[self.target])

        # Prediction ------------------------------------------------------------------------------
        self.pred_test_file = self.split_test_file
        self.pred_max_lengths = {
            "SMILES": 128,
            "SIGNATURE": 3072,
            "ECFP": 512,
        }
        if self.target is not None:
            self.pred_max_length = self.pred_max_lengths[self.target]
        else:
            self.pred_max_length = None

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "Config":
        """Return a Config object from a preset."""
        presets = {
            "emolecules": {
                "mode": "translate",
                "db": "emolecules",
                "split_method": "train_valid_split",
            },
            "metanetx": {
                "mode": "translation",
                "db": "metanetx",
                "split_method": "kfold",
                "split_num_folds": 5,
            },
        }
        # presets = {
        #     "ECFP -> SMILES": {
        #         "mode": "translate",
        #         "source": "ECFP",
        #         "target": "SMILES"
        #     },
        #     "SIGNATURE -> SMILES": {
        #         "mode": "translate",
        #         "source": "SIGNATURE",
        #         "target": "SMILES"
        #     },
        # }
        if preset not in presets:
            raise ValueError(f"Preset '{preset}' not found")
        params = presets[preset]
        params.update(kwargs)  # Overload default parameters
        return cls(**params)

    def with_base_dir(self, base_dir: Path | str) -> "Config":
        """Return a new Config object with the base_dir attribute updated."""
        base_dir = Path(base_dir).resolve()
        new_config = replace(self, base_dir=Path(base_dir))
        new_config.__post_init__()
        return new_config

    def with_db(self, db: str) -> "Config":
        """Return a new Config object with the db attribute updated."""
        new_config = replace(self, db=db)
        new_config.__post_init__()
        return new_config

    def with_split_method(self, split_method: str) -> "Config":
        """Return a new Config object with the split_method attribute updated."""
        new_config = replace(self, split_method=split_method)
        new_config.__post_init__()
        return new_config

    def with_fold_index(self, train_fold_index: int) -> "Config":
        """Return a new Config object with the train_fold_index attribute updated."""
        new_config = replace(self, train_fold_index=train_fold_index)
        new_config.__post_init__()
        return new_config

    def with_source(self, source: str) -> "Config":
        """Return a new Config object with the source attribute updated."""
        new_config = replace(self, source=source)
        new_config.__post_init__()
        return new_config

    def with_target(self, target: str) -> "Config":
        """Return a new Config object with the target attribute updated."""
        new_config = replace(self, target=target)
        new_config.__post_init__()
        return new_config

    def with_token_dir(self, token_dir: str) -> "Config":
        """Return a new Config object with the token_dir attribute updated."""
        new_config = replace(self)
        new_config.token_dir = Path(token_dir).resolve()
        new_config.token_models = {
            "SMILES": new_config.token_dir / "SMILES.model",
            "SIGNATURE": new_config.token_dir / "SIGNATURE.model",
            "ECFP": new_config.token_dir / "ECFP.model",
        }
        return new_config

    def to_dict(self) -> dict:
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif is_dataclass(obj):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            else:
                return obj

        return serialize(self)
