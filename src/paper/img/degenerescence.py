# Load file and sanitize molecule

import argparse
import collections
import csv
import glob
import gzip
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

def handler(path):
    hd = None
    if path.endswith("gz"):
        hd = gzip.open(path, "rt", newline="")
    else:
        hd = open(path, newline="")
    return hd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dataset-csv",
        required=True,
        help="Path of the dataset, .csv or .csv.gz",
    )
    parser.add_argument(
        "--output-dataset-json",
        required=True,
        help="Path of the dataset json",
    )

    args = parser.parse_args()

    path = args.input_dataset_csv
    # Parse data
    cols = ["SIG", "SIG-NEIGH", "SIG-NBIT", "SIG-NEIGH-NBIT", "SMILES", "ECFP4"]
    stats = {}

    total = 0
    for ix, col in enumerate(cols):
        # Parse file
        counter = collections.Counter()
        csvfile = handler(path=path)
        reader = csv.DictReader(csvfile)
        for row in reader:
            counter.update([row[col]])
        if ix < 1:
            stats["total"] = reader.line_num - 1 # rm header
        csvfile.close()
        # Get data
        collision = collections.Counter()
        for value in counter.values():
            collision.update([value])
        stats[col] = dict(collision)
        del counter, collision

    # Save
    with open(args.output_dataset_json, "w") as fd:
        json.dump(stats, fd)
