import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_df(path: str, src: str) -> pd.DataFrame:
    data = json.load(open(path))
    del data["total"]
    df_raw = pd.DataFrame(data)

    # Rename cols
    for col in df_raw.columns:
        ncol = col.replace("_", "-").upper()
        df_raw.rename(columns={col: ncol}, inplace=True)

    df = pd.DataFrame()
    for col in df_raw.columns:
        df_col = df_raw[col].to_frame()
        df_col["label"] = col
        df_col.rename(columns={col: "count"}, inplace=True)
        df = pd.concat([df, df_col])

    # Create "duplicate"
    df.reset_index(inplace=True)
    df.rename(columns={"index": "duplicate"}, inplace=True)

    # Fmt
    df.fillna(0, inplace=True)
    df["duplicate"] = df["duplicate"].astype(int)
    df["count"] = df["count"].astype(int)
    df["count_log"] = df["count"].apply(lambda x: math.log(x) if x > 0 else 0)

    df["src"] = src

    df = df[df["label"] != "SMILES"]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-metanetx-json",
        help="Path of the dataset, .json",
    )
    parser.add_argument(
        "--input-emolecules-json",
        help="Path of the dataset, .json",
    )
    parser.add_argument(
        "--output-graph-png",
        help="Path of the dataset, .png",
    )
    parser.add_argument(
        "--output-graph-csv",
        help="Path of the dataset, .csv",
    )

    args = parser.parse_args()

    fgraph = args.output_graph_png
    fcsv = args.output_graph_csv

    # Load data
    df_metanetx = create_df(path=args.input_metanetx_json, src="metanetx")
    df_emolecules = create_df(path=args.input_emolecules_json, src="emolecules")

    df = pd.concat([df_metanetx, df_emolecules])
    df.sort_values("duplicate", inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Graph
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(data=df, x="duplicate", y="count_log", hue="src", style="label")
    ax.set(xlabel="Duplicate", ylabel="Occurence (log)")

    if fgraph:
        plt.savefig(fgraph)
    if fcsv:
        df.to_csv(fcsv, index=False)
