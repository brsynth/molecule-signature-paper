# Load file and sanitize molecule

import argparse
import collections
import csv
import glob
import gzip
import json
import os
import sqlite3
import time
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
        help="Path of the dataset, .csv or .csv.gz",
    )
    parser.add_argument(
        "--input-dataset-sql",
        help="Path of the dataset, .sql",
    )

    parser.add_argument(
        "--output-dataset-json",
        required=True,
        help="Path of the dataset json",
    )

    args = parser.parse_args()

    # Parse data
    cols = ["SIG", "SIG-NEIGH", "SIG-NBIT", "SIG-NEIGH-NBIT", "SMILES", "ECFP4"]
    stats = {}

    if args.input_dataset_csv:
        path = args.input_dataset_csv
        total = 0
        for ix, col in enumerate(cols):
            # Parse file
            counter = collections.Counter()
            csvfile = handler(path=path)
            reader = csv.DictReader(csvfile)
            for row in reader:
                counter.update([row[col]])
            if ix < 1:
                stats["total"] = reader.line_num - 1  # rm header
            csvfile.close()
            # Get data
            collision = collections.Counter()
            for value in counter.values():
                collision.update([value])
            stats[col] = dict(collision)
            del counter, collision
    if args.input_dataset_sql:
        path = args.input_dataset_sql
        print("Deal with:", path)
        total = 0
        for ix, col in enumerate(cols):
            print("Analyze:", col)
            col = col.replace("-", "_").lower()

            # Parse file
            collision = collections.Counter()

            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            print("Query - start")
            start = time.time()
            cursor.execute("SELECT COUNT(*) FROM compound GROUP BY %s" % (col,))
            for res in cursor.fetchall():
                collision.update([res[0]])
            print("Query - end", time.time() - start)
            stats[col] = dict(collision)

            # Get total
            if ix < 1:
                cursor.execute("SELECT COUNT(%s) FROM compound" % (col,))
                res = cursor.fetchone()
                if res:
                    stats["total"] = res[0]

            del collision

    # Save
    with open(args.output_dataset_json, "w") as fd:
        json.dump(stats, fd)
