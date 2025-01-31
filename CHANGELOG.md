## 3.0.0 (2025-01-31)

### Feat

- **predict**: predict call from CLI now always outputs something
- **2.enumeration_results**: notebook to perform the molecule enumeration
- **1.enumeration_create_alphabets**: notebook to create alphabets
- **notebooks**: add  notebook
- **configure**: add interface for simple configuration
- **predict**: add interface for predicting (without evaluation)
- **model**: fully batch-vectorized version for beam search
- **predict**: test for equality of canonic SMILES
- **predict**: decode beam in parallel

### Fix

- **predict**: convert python objects to strings
- **predict**: dataclass attribute poorly tested
- **predict**: remove pickle writing
- **evaluate**: add output file handling to write results
- **evaluate**: correctly deal with the max number of rows
- **evaluate**: include chirality in ECFP
- **predict**: stop crashing when beam size > vocab size
- **predict**: allow selection of the accelerator device
- **predict**: column indexes

### Refactor

- **predict**: update default arg values
- **predict**: update result refinements
- **configure**: add default output path to None
- **config**: remove unused method
- **imports**: refine
- add empty data folder
- **notebooks**: merge cells
- **notebooks**: rename nb
- **notebooks**: rename notebook for fig 2
- **utils**: move utilities functions
- **evaluate**: remove file
- **predict**: make col names more explicit
- **evaluate**: sweep code
- **utils**: sweep code
- **predict**: refine outputs
- **predict**: remove unused args
- **utils**: additional shared functions
- **predict**: delagate results refining to subsequent code
- **predict**: allow calls from other script
- **predict**: better print
- **predict**: improve imports
- **predict**: print result to stdout on request
- **predict**: print default values

## 2.1.0 (2025-01-07)

### Feat

- **prepare**: outputs additional column signature_morgans

### Fix

- **prepare**: remove deprecated import
- **get_smiles**: remove superflous Hs
- **prepare**: sanitize molecule after stereo-isomer enumeration
- **prepare**: add missing header

### Refactor

- remove old code
- **.env**: ignore local env file
- erase old code

## 2.0.4 (2024-12-10)

### Fix

- update changelog on version bump

## 2.0.3 (2024-12-10)

### Fix

- attempt to trigger GA

## 2.0.2 (2024-12-10)

### Fix

- main instead of master branch name

## 2.0.1 (2024-12-10)

### Fix

- **dataset**: remove unused code

## 2.0.0 (2024-12-09)

### Feat

- **learning**: add transformer code
- **dataset**: add code to compute model tokens
- **dataset**: add code for download and prepare datasets
- **transformer/train**: additional arg for setting source / target max length
- **transformer/train**: implement gradient accumulation
- **transformer/train**: define num of data loader workers from args
- **transformer/train**: make modele compilation by Torch optional
- **transformer/train**: generalize mixed precision scaler usage
- **transformer/model**: refine state_dict Module's method
- **transformer/train**: check for NaNs in loss
- **transformer/train**: model dir output as arg
- **transformer/train**: experimentation with mixed precision floats
- **transformer/train**: make use of pin_memory=true in dataloaders expected to increase GPU perf
- **transformer/train**: first working version
- **transformer**: in dev code
- new code to download and make use of the signature code (#10)

### Fix

- **transformer/train**: load_checkpoint
- **transformer/train**: effective batch indexes
- **transformer/train**: duplicated loss normalization
- **transformer/train**: wrong arg name
- **transformer/train**: take into account remaining remaining batches for the sceduler counts
- **transformer/train**: propagate gradient for last batches of epoch
- **transformer/train**: remove multiple calls to unscale_
- **transformer/train**: use save_checkpoint
- **transformer/train**: refine save and load methods
- **transformer/train**: correct seq length arg
- **transformer/train**: stop sending to preset device
- **dataset/utils.py**: forward pass logger in recursive calls
- **tokenizer**: allow additional depictions

### Refactor

- **transformer**: sweep code
- **dataset**: clean deprecated code
- **transformer**: remove deprecated code
- **transformer/train**: refine gradient accumulation
- **transformer/config**: reduce learning rate to prevent NaN / Inf values
- **transformer/train**: make GPU pinned memory an option
- **transformer/train**: add few debug messages
- **transformer/config**: update
- **transformer/config**: update
- **transformer/train**: get the number of epochs from config
- **transformer/train**: better log Nan / Inf value issues
- **transformer/config**: increase learning rate
- **transformer/config**: increase learning rate
- **transformer/config**: reduce learning rate
- **transformer/train**: update default log level
- **transformer/train**: better handle device arg
- **transformer/config.yaml**: update training values
- **model**: remove unecessary code
- **dataset/utils.py**: don't sort config keys
- **download**: update paths

### Perf

- **transformer/train**: AdamW optimizer instead of Adam, OneCycleLR scheduler

## 1.1.0 (2023-10-30)

### Feat

- **download_metanetx**: generate sig alphabet with nbit and neighbors
- **tokenizer**: refactor and enable unigram model type
- **tokenizer**: new arguments to select tokenizer model, depic to treat and pairs to build
- **tokenizer**: increase script verbosity
- **tokenizer**: use all tokens available and support unigram model
- **tokenize**: write SIG-NEIGH-NBIT datasets
- **tokenizer**: produce SIG-NEIGH-NBIT datasets
- **paper**: improve speed for emolecules
- **paper**: img, add
- **paper**: enable sig-nbit
- **paper**: construct alphabet for sig-nbit
- **paper**: download, add emolecules
- **paper**: img, add degenerescence
- **library**: update to RevSig1.5
- **paper**: download, enable formalCharge in sanitize
- **paper**: tokenizer, use ECFP4_COUNT
- **paper**: download, add FP count and extract test_small

### Fix

- **tokenizer**: fix regular expression
- **download_metanetx**: fix paths
- **download_metanetx**: fix paths
- **signature**: use ECFP instead of FCFP
- **tokenizer**: stop spliting SIG bond tokens
- **tokenizer**: spelling in AROMATIC bond regex
- **tokenizer**: stop omitting bounds in regex
- **paper**: dataset, ecfp4 duplicate index number according to the count
- **paper**: tokenizer, use the right function

### Refactor

- **download_metanetx**: progress bar and more logs
- **download_metanetx**: print settings
- **download_metanetx**: store file paths in args.dict

## 1.0.0 (2023-08-09)

### Feat

- **download**: introduce default output dir
- **tokenizer**: build vocabularies and dataset pairs
- **tokenizer**: only output on-bits in ECFP4
- **tokenizer**: add sentencepiece tokenizer
- **retrosig**: add utils/cmd.py file
- **library**: update with "RevSig1.2"

### Fix

- **tokenizer**: fingerprints name in upper case to match expectation
- **download**: put back right path for rdkit method (#7)
- **download**: fix argparse crash due to percent sign in help (#6)
- **download**: shuffle data only once
- **download**: prevent removing raw mnx file
- **download**: create ouput dir if it not exists

### Refactor

- **tokenizer**: change file pairs extension
- **download**: simplify args usage
- sweep imports
- **download**: pointing out unexpected filtered smiles
- **download**: update ouput name for the signature alphabet file
- **download**: change default value of test and valid datasets
- **download**: disable shuffling before sanitizing

## 0.1.0 (2023-07-27)

### Feat

- **paper**: add tokenizer signature
- **alphabet**: build reaction

## 0.0.1 (2023-07-27)
