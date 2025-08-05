#!/usr/bin/env python3.9
# coding=utf-8

"""
Display basic road accident statistics based on different factors.

Student: Jan Matufka <xmatuf00@stud.fit.vutbr.cz>
IZV Course Assignment #2
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# muzete pridat libovolnou zakladni knihovnu
# ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

""" Ukol 1:
načíst soubor nehod, který byl vytvořen z vašich dat.
Neznámé integerové hodnoty byly mapovány na -1.

Úkoly:

- vytvořte sloupec date, který bude ve formátu data
(berte v potaz pouze datum, tj sloupec p2a)

- vhodné sloupce zmenšete pomocí kategorických datových typů.
Měli byste se dostat po 0.5 GB.
Neměňte však na kategorický typ region
(špatně by se vám pracovalo s figure-level funkcemi)

- implementujte funkci, která vypíše kompletní (hlubkou)
velikost všech sloupců v DataFrame v paměti:
orig_size=X MB
new_size=X MB

Poznámka: zobrazujte na 1 desetinné místo (.1f) a počítejte, že 1 MB = 1e6 B.
"""

regions = ["HKK", "JHC", "JHM", "KVK"]
MiB_size = 1_048_576


# Ukol1: Predpříprava dat
def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and prepare a locally stored file with acciedent stats for analysis.

    Filepath is specified by filename argument.
    Verbose is a flag for printing extra compression info.
    Function performs intger and float compression where possible.
    """
    df = pd.read_pickle(filename)

    if verbose:
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


# Ukol 2: Počty nehod v jednotlivých regionech podle druhu silnic
def plot_roadtype(df: pd.DataFrame, fig_location: str = None,
                  show_figure: bool = False):
    """
    Create graphs with road accident statistics based on roadtype.

    Roadtype is distinguished by the amount of lanes.
    Each roadtype is shown on a separatele graph.
    """
    data = df[df["region"].isin(regions)]
    data = data[["region", "p21"]]
    data["count"] = 1
    data = data.sort_values(by="p21")

    data["p21"] = data["p21"].map({
        0: "Žiadna",
        1: "Dvojpruhová",
        2: "Trojpruhová",
        3: "Štvorpruhová",
        4: "Štvorpruhová",
        5: "Viacpruhová",
        6: "Rýchlostná komunikácia",
    })

    data = data.groupby(by=["p21", "region"], sort=False).sum().reset_index()

    plot = sns.catplot(
        data=data,
        kind="bar",
        x="region",
        y="count",
        ci=None,
        col="p21",
        col_wrap=3,
        sharex=True,
        sharey=False,
        palette='Accent',
    )

    plot.fig.suptitle("Počet nehôd podľa druhu cesty")
    plot.set_axis_labels("Kraj", "Počet nehôd")
    plot.set_titles("Typ vozovky: {col_name}")
    plot.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


# Ukol3: zavinění zvěří
def plot_animals(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Create graphs with road accident statistics based on whose fault was it.

    Only analyses two culprits: animals / humans.
    Other culprits are aggregated into another category.
    Each region is shown on a separatele graph.
    """
    columns = ["region", "p10", "date"]

    data = df[(df["region"].isin(regions)) & (
        df["p58"] == 5) & (df["date"].dt.year < 2021)]
    data = data[columns]
    data["count"] = 1
    data["date"] = data["date"].dt.month
    data = data.sort_values(by="date")

    data["p10"] = data["p10"].map({
        1: "Vodičom",
        2: "Vodičom",
        4: "Zverou"}).fillna("Iné")

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

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Máj', 'Jún', 'Júl',
              'Aug',  'Sep',  'Okt',  'Nov',  'Dec']

    plot.set_xticklabels(months)
    plot.fig.suptitle("Počet nehôd podľa vinníka")
    plot.set_axis_labels("", "Počet nehôd")
    plot.set_titles("Kraj: {col_name}")
    plot.tight_layout()
    plot._legend.set_title("Zavinenie")

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


# Ukol 4: Povětrnostní podmínky
def plot_conditions(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    Create graphs for accident statistics based on weather conditions.

    Distinguishes 7 different conditions.
    Each region is shown on a separatele graph.
    """
    data = df[(df["region"].isin(regions)) & (df["p18"] != 0)]
    data = data[["region", "p18", "date"]]
    data["count"] = 1
    data = data.sort_values(by="date")

    data["p18"] = data["p18"].map({
        1: "Nesťažené",
        2: "Hmla",
        3: "Začiatok dažďa",
        4: "Dážď",
        5: "Sneženie",
        6: "Námraza",
        7: "Vietor",
    })

    data = pd.pivot_table(
        data,
        columns="p18",
        values="count",
        index=["region", "date"],
        aggfunc="sum"
    ).fillna(0)

    data = data.stack(level=["p18"])
    data = data.to_frame().reset_index()
    data = data.groupby(["region", "p18"]).resample(
        'M', on="date").sum().reset_index()

    plot = sns.relplot(
        data=data,
        kind="line",
        x="date",
        y=0,
        hue="p18",
        col="region",
        col_wrap=2,
        facet_kws={"sharey": False, "sharex": False},
        palette='Accent',
    )

    plot.fig.suptitle("Nehodovosť v čase v rôznom počasí pre jednotlivé kraje")
    plot.set(xlim=(pd.to_datetime("1/1/2016"), pd.to_datetime("1/1/2021")))
    plot.set_titles("Kraj: {col_name}")
    plot.set_axis_labels("Rok", "Počet nehôd")
    plot._legend.set_title("Počasie")
    plot.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkretni
    # funkce.

    # tento soubor si stahnete sami, při testování pro hodnocení bude existovat
    df = get_dataframe("accidents.pkl.gz", True)
    plot_roadtype(df, fig_location="01_roadtype.png", show_figure=False)
    plot_animals(df, "02_animals.png", False)
    plot_conditions(df, "03_conditions.png", False)
