import pandas as pd


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
            "Query ECFP",
            "SMILES Exact Match",
            "Tanimoto Exact Match",
            "SMILES Syntaxically Valid",
            "Tanimoto Exact Match Unique Count",
            "Tanimoto Exact Match Unique List",
        ],
        index=query_ids,
    )

    # Now for can collect results for query ID
    for query_id in query_ids:

        # Get mask corresponding to the query ID
        query_mask = results["Query ID"] == query_id

        # Get subset corresponding to the top-k
        top_query_subset = results[query_mask].nlargest(topk, "Predicted Log Prob")

        # Get the subset from the top-k corresponding to Tanimoto exact match
        top_query_exact_match = top_query_subset[top_query_subset["Tanimoto Exact Match"]]

        # Fill in the stats
        summary.loc[query_id, "Query ID"] = query_id
        summary.loc[query_id, "Query SMILES"] = top_query_subset.iloc[0]["Query SMILES"]
        summary.loc[query_id, "Query ECFP"] = top_query_subset.iloc[0]["Query ECFP"]
        summary.loc[query_id, "SMILES Exact Match"] = any(top_query_subset["SMILES Exact Match"])
        summary.loc[query_id, "Tanimoto Exact Match"] = any(
            top_query_subset["Tanimoto Exact Match"]
        )
        summary.loc[query_id, "SMILES Syntaxically Valid"] = any(
            top_query_subset["SMILES Syntaxically Valid"]
        )
        summary.loc[query_id, "Tanimoto Exact Match Unique Count"] = top_query_exact_match[
            "Predicted Canonic SMILES"
        ].nunique()
        summary.loc[query_id, "Tanimoto Exact Match Unique List"] = str(
            list(top_query_exact_match["Predicted Canonic SMILES"].unique())
        )

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
    stats = summary.aggregate(
        {
            "SMILES Exact Match": ["mean"],
            "Tanimoto Exact Match": ["mean"],
            "SMILES Syntaxically Valid": ["mean"],
        }
    )

    # Rename columns
    stats.columns = [
        "SMILES Accuracy",
        "Tanimoto Accuracy",
        "SMILES Syntax Validity",
    ]

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
