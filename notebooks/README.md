# Notebooks

## 4.degeneracy

This notebook analyzes the degeneracy of Signatures and ECFP4 starting from SMILES molecules found in eMolecules and MetaNetX.  
Signatures and ECFP4 are computed for each molecule and stored in a `sqlite` database.
Two tables are created `metanetx` and `emolecules`, each with these columns: `smiles`, `ecfp4`, `sig`, `sig_nei`.
Degeneracy is computed across the sql database.
Note that this script takes approximately 24 hours to complete using 8 threads and 20G of memory.  

It requires the following supplementary dependencies: `ipywidgets`, `openbabel`, `pandarallel`, `seaborn`, and optionally `papermill` for command-line execution.
Parameters in cells tagged as `parameters` can be adjusted to meet your specific needs.  
