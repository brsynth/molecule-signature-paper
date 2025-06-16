"""Handy functions for analyzing results from the notebook."""

import time

import pandas as pd

from MolForge.fingerprints import ECFP4


def mol_to_ecfp_molforge(mol):
    """Convert a molecule to its ECFP4 fingerprint as used in MolForge."""
    if mol is None:
        return None
    return ECFP4(mol)


def ecfp_to_string_molforge(ecfp):
    """Convert an ECFP4 fingerprint to a string representation as used in MolForge."""
    if ecfp is None:
        return None
    return " ".join(str(_) for _ in list(ecfp.GetNonzeroElements()))


def get_summary(results: pd.DataFrame, topk=1) -> pd.DataFrame:
    """Get summary from the results DataFrame.

    Parameters
    ----------
    results : pandas.DataFrame
        The results DataFrame.
    topk : int, optional
        The top-k to consider, by default 1.

    Returns
    -------
    pandas.DataFrame
        The summary DataFrame.
    """

    # First we need to get the unique sequence IDs
    query_ids = results["Query ID"].unique()

    summary = pd.DataFrame(
        columns=[
            "Query ID",
            "Query SMILES",
            "Query Counted ECFP",
            "Query Binary ECFP",
            "SMILES Exact Match",
            "SMILES Exact Match No Stereo",
            "Tanimoto Counted ECFP Exact Match",
            "Tanimoto Counted ECFP Exact Match No Stereo",
            "Tanimoto Binary ECFP Exact Match",
            "SMILES Syntaxically Valid",
            "Tanimoto Exact Match Unique Count",
            "Tanimoto Exact Match Unique List",
            "Time Elapsed",
        ],
        index=query_ids,
    )

    # Remove "Binary" columns if they are not present in the results
    if "Query Binary ECFP" not in results.columns:
        summary.drop(columns=["Query Binary ECFP"], inplace=True)
    if "Tanimoto Binary ECFP Exact Match" not in results.columns:
        summary.drop(columns=["Tanimoto Binary ECFP Exact Match"], inplace=True)

    # Remove "no stereo" columns if they are not present in the results
    if "SMILES Exact Match No Stereo" not in results.columns:
        summary.drop(columns=["SMILES Exact Match No Stereo"], inplace=True)
    if "Tanimoto Counted ECFP Exact Match No Stereo" not in results.columns:
        summary.drop(columns=["Tanimoto Counted ECFP Exact Match No Stereo"], inplace=True)

    # Remove Tim Elapsed column if it is not present in the results
    if "Time Elapsed" not in results.columns:
        summary.drop(columns=["Time Elapsed"], inplace=True)

    # Now for can collect results for query ID
    for query_id in query_ids:

        # Get mask corresponding to the query ID
        query_mask = results["Query ID"] == query_id

        # Get subset corresponding to the top-k
        top_query_subset = results[query_mask].nlargest(topk, "Predicted Log Prob")

        # Get the subset from the top-k corresponding to Tanimoto exact match
        top_query_exact_match = top_query_subset[top_query_subset["Tanimoto Counted ECFP Exact Match"]]

        # Fill in the stats
        summary.loc[query_id, "Query ID"] = query_id
        summary.loc[query_id, "Query SMILES"] = top_query_subset.iloc[0]["Query SMILES"]
        summary.loc[query_id, "Query Counted ECFP"] = top_query_subset.iloc[0]["Query Counted ECFP"]

        # Check if at least one of the top-k corresponds to the query SMILES (with and without stereo)
        summary.loc[query_id, "SMILES Exact Match"] = any(top_query_subset["SMILES Exact Match"])
        if "SMILES Exact Match No Stereo" in top_query_subset.columns:
            summary.loc[query_id, "SMILES Exact Match No Stereo"] = any(top_query_subset["SMILES Exact Match No Stereo"])  # noqa: E501

        # Check if at least one of the top-k has corresponds to a Tanimoto exact match (with and without stereo, and MolForge)  # noqa: E501
        summary.loc[query_id, "Tanimoto Counted ECFP Exact Match"] = any(top_query_subset["Tanimoto Counted ECFP Exact Match"])
        if "Tanimoto Counted ECFP Exact Match No Stereo" in top_query_subset.columns:
            summary.loc[query_id, "Tanimoto Counted ECFP Exact Match No Stereo"] = any(top_query_subset["Tanimoto Counted ECFP Exact Match No Stereo"])  # noqa: E501
        if "Tanimoto Binary ECFP Exact Match" in top_query_subset.columns:
            summary.loc[query_id, "Tanimoto Binary ECFP Exact Match"] = any(top_query_subset["Tanimoto Binary ECFP Exact Match"])

        # Check if at least one of the top-k has a valid SMILES
        summary.loc[query_id, "SMILES Syntaxically Valid"] = any(top_query_subset["SMILES Syntaxically Valid"])  # noqa: E501

        # Count the number of unique SMILES in the Tanimoto exact match subset
        summary.loc[query_id, "Tanimoto Exact Match Unique Count"] = top_query_exact_match["Predicted Canonic SMILES"].nunique()  # noqa: E501
        summary.loc[query_id, "Tanimoto Exact Match Unique List"] = str(list(top_query_exact_match["Predicted Canonic SMILES"].unique()))  # noqa: E501

        # If time elapsed is present, fill it in
        if "Time Elapsed" in top_query_subset.columns:
            summary.loc[query_id, "Time Elapsed"] = top_query_subset.iloc[0]["Time Elapsed"]

    return summary


def get_statistics(df: pd.DataFrame, topk=1) -> pd.DataFrame:
    """Get statistics from the results DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The results DataFrame.
    topk : int, optional
        The top-k to consider, by default 1.

    Returns
    -------
    pandas.DataFrame
        The statistics DataFrame.
    """

    # First we get summary information
    summary = get_summary(df, topk=topk)

    # Now we can compute basic statistics
    instructions = {
        "SMILES Exact Match": ["mean"],
        "Tanimoto Counted ECFP Exact Match": ["mean"],
        "SMILES Syntaxically Valid": ["mean"],
    }

    # No stereo columns are optional, so we check if they are present
    if "SMILES Exact Match No Stereo" in summary.columns:
        instructions["SMILES Exact Match No Stereo"] = ["mean"]
    if "Tanimoto Counted ECFP Exact Match No Stereo" in summary.columns:
        instructions["Tanimoto Counted ECFP Exact Match No Stereo"] = ["mean"]

    # MolForge columns are also optional
    if "Tanimoto Binary ECFP Exact Match" in summary.columns:
        instructions["Tanimoto Binary ECFP Exact Match"] = ["mean"]

    # Time Elapsed is also optional
    if "Time Elapsed" in summary.columns:
        instructions["Time Elapsed"] = ["mean"]

    stats = summary.aggregate(instructions)

    # Rename columns
    stats.rename(
        columns={
            "SMILES Exact Match": "SMILES Accuracy",
            "SMILES Exact Match No Stereo": "SMILES No Stereo Accuracy",
            "Tanimoto Counted ECFP Exact Match": "Tanimoto Counted ECFP Accuracy",
            "Tanimoto Counted ECFP Exact Match No Stereo": "Tanimoto Counted ECFP No Stereo Accuracy",
            "Tanimoto Binary ECFP Exact Match": "Tanimoto Binary ECFP Accuracy",
            "SMILES Syntaxically Valid": "SMILES Syntax Validity",
            "Time Elapsed": "Average Time Elapsed",
        },
        inplace=True,
    )

    # Transpose and set index as a "Stat" column
    stats = stats.T
    stats["Stat"] = stats.index
    stats.reset_index(drop=True, inplace=True)

    # Rename and reorder columns
    stats.columns = ["Value", "Stat"]
    stats = stats[["Stat", "Value"]]

    return stats


def get_uniqueness(df: pd.DataFrame, topk=1) -> pd.DataFrame:
    """Get the number of unique molecules per query.

    Parameters
    ----------
    df : pandas.DataFrame
        The results DataFrame.
    topk : int, optional
        The top-k to consider, by default 1.

    Returns
    -------
    pandas.DataFrame
        The unique count per query DataFrame.
    """
    # First we get summary information
    summary = get_summary(df, topk=topk)

    # Get count on the number of unique SMILES per query
    uniqueness = pd.DataFrame(
        summary["Tanimoto Exact Match Unique Count"].value_counts().sort_index()
    )
    uniqueness.rename(columns={"count": "Count"}, inplace=True)
    uniqueness["Distinct Molecules per Query"] = uniqueness.index
    uniqueness.reset_index(drop=True, inplace=True)
    uniqueness = uniqueness.iloc[:, [1, 0]]  # reverse the order of the columns

    return uniqueness


class Timer:
    """A simple timer class to measure execution time."""
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self  # permet d'utiliser "as" dans le with

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
