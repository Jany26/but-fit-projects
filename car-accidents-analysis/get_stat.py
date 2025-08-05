#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creating some graphs using numpy and matplotlib.

1st part of a project for IZV course (FIT BUT).
Student Login:      xmatuf00
Student Full name:  Ján Maťufka
File:               get_stat.py
"""

# povolene jsou pouze zakladni knihovny (os, sys)
# a knihovny numpy, matplotlib a argparse
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import argparse
from download import DataDownloader


def parseargs():
    """Command line argument parsing."""
    desc = """Shows some accident stats scraped from CZ police pages.
    Part of a project for IZV Course.
    (Data processing and visualization in Python)
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--fig_location',
                        type=str,
                        required=False,
                        help='Path to a file to store the statistics image.')
    parser.add_argument('--show_figure',
                        action='store_true',
                        required=False,
                        help='Program will show graphs in a window.')
    args = parser.parse_args()
    if args.fig_location is not None:
        directory = os.path.dirname(args.fig_location)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)
        name = os.path.basename(args.fig_location)
        if not name.endswith(".png"):
            raise Exception("Unsupported image format")
    return args


def plot_stat(data_source, fig_location=None, show_figure=False):
    """
    Create 2 basic graphs with matplotlib.

    Graphs show accident statics for each region
    based on road conditions.
    """
    if fig_location is None and show_figure is False:
        return
    data = data_source
    sorted_regions = sorted(data.regions)

    absolutes = []
    for region in sorted_regions:
        counts = [np.count_nonzero(data.data[region]['p24'] == j)
                  for j in range(6)]
        absolutes.append(counts)

    absolutes = np.transpose(np.array(absolutes))
    absolutes = np.roll(absolutes, shift=-1, axis=0)
    relatives = []
    for row in absolutes:
        relatives.append(row / np.sum(row) * 100)
    relatives = np.array(relatives)
    relatives[relatives == 0.0] = np.nan

    values_desc = [
        "Přerušovaná žlutá",
        "Semafor mimo provoz",
        "Dopravními značky",
        "Přenosné dopravní značky",
        "Nevyznačená",
        "Žádna úprava",
    ]

    plt.rcParams.update({'font.size': 8})
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.tight_layout(rect=(0.2, 0, 1, 1))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.45)
    fig.dpi = 100

    # first graph
    pcm = ax0.imshow(absolutes, cmap='viridis',
                     norm=LogNorm(vmax=10**5, vmin=10**0))
    ax0.title.set_text('Absolutně')

    ax0.set_xticks(range(14))
    ax0.set_xticklabels([sorted_regions[i] for i in range(14)])
    ax0.set_yticks(range(6))
    ax0.set_yticklabels([values_desc[i] for i in range(6)])

    cbar = fig.colorbar(pcm, ax=ax0, shrink=1.1)
    cbar.set_ticks([10**i for i in range(6)])
    cbar.set_label('Počet nehod', rotation=90)

    # second graph
    pcm = ax1.imshow(relatives, cmap='plasma', vmax=100, vmin=0)
    ax1.title.set_text('Relativně vůči přičině')

    ax1.set_xticks(range(14))
    ax1.set_xticklabels([sorted_regions[i] for i in range(14)])
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([values_desc[i] for i in range(6)])

    cbar = fig.colorbar(pcm, ax=ax1, shrink=1.1)
    cbar.set_ticks([20*i for i in range(6)])
    cbar.set_label('Počet nehod pro danou příčinu [%]', rotation=90)

    if fig_location is not None:
        plt.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        plt.show()


# TODO pri spusteni zpracovat argumenty
def main():
    """Get data and call plotting function."""
    args = parseargs()
    data = DataDownloader()
    data.get_dict()
    plot_stat(data, args.fig_location, args.show_figure)


if __name__ == '__main__':
    main()

# End of file get_stat.py
