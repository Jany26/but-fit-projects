#!/usr/bin/python3.8
# coding=utf-8

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import seaborn as sns
import contextily
import sklearn.cluster
import numpy as np

def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and prepare a locally stored file with acciedent stats for analysis.

    Filepath is specified by filename argument.
    Verbose is a flag for printing extra compression info.
    Function performs intger and float compression where possible.
    """
    df = pd.read_pickle(filename)

    if verbose:
        MiB_size = 1_048_576
        orig_size = df.memory_usage(deep=True).sum()

    df["date"] = df["p2a"].astype("datetime64")
    df["p2a"] = df["p2a"].astype("datetime64")

    for col in df.columns:
        if col in {"region", "date", "p2a"}:
            continue

        if df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast='signed')
        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = df[col].astype("category")

    if verbose:
        print(f"orig_size={orig_size/MiB_size:.1f} MB")
        print(f"new_size={df.memory_usage(deep=True).sum()/MiB_size:.1f} MB")

    return df


def plot_roadstate(df: pd.DataFrame):
    columns = ["region", "p17"]
    data = df[(df["p17"] != 1)]
    data = data["p17"] = data["p17"].map({
        2: "Sklon vyšší ako 8 %",
        3: "Problém so značkou (zle umiestnená, znečistená, resp. chýbajúca)",
        4: "Zvlnený povrch v pozdĺžnom smere",
        5: "Súvislé výtlky",
        6: "Nesúvislé výtlky",
        7: "Trvalé zúženie vozovky",
        8: "Priečna stružka, hrbol, vystuplé/prepadlé koľaje",
        9: "Neoznačená/Zle označená prekážka na ceste",
        10: "Prechodná uzávierka 1 jazdného pruhu",
        11: "Prechodná uzávierka cesty alebo jazdného pásu",
        12: "Iné / Neuvedené"
    })
    print(data)

    data = data.groupby(by=columns, sort=False).sum().reset_index()

    plot = sns.catplot(
        data=data,
        kind="bar",
        x="date",
        y="count",
        hue="p10",
        ci=None,
        col="region",
        col_wrap=2,
        sharex=False,
        sharey=False,
        palette='Accent',
    )
    plt.show()
    pass

def table_roadstate():
    pass

def calculate_values():
    pass

if __name__ == '__main__':
    df = get_dataframe("accidents.pkl.gz", True)
    plot_roadstate(df)
    pass