#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downloading, parsing and cleaning CSV data to be processed.

1st part of a project for IZV course (FIT BUT).
Student Login:      xmatuf00
Student Full name:  Ján Maťufka
File:               download.py
"""

from bs4 import BeautifulSoup
import numpy as np

import io
import os
import gzip
import requests
import pickle
import zipfile
import csv

# Kromě vestavěných knihoven (os, sys, re, requests …)
# byste si měli vystačit s:
# gzip, pickle, csv, zipfile, numpy, matplotlib, BeautifulSoup.
# Další knihovny je možné použít po schválení opravujícím (např ve fóru WIS).


class DataDownloader:
    """
    Class handles data downloading, csv file parsing and caching.

    Class Attributes:
        headers     Nazvy hlavicek jednotlivych CSV souboru,
                    tyto nazvy nemente!
        regions     Dictionary s nazvy kraju : nazev csv souboru
        types       Numpy types based on header names

    Instance Attributes:
        url             From which website will the files be downloaded
                        (works with only one web)
        folder          Folder for storing downloaded and cache files
        cache_filename  Name format for cache files
        zipfiles        Array of filenames that are downloaded (for checking)
        data            Dictionary of all parsed data ()
    """

    headers = [
        "p1", "p36", "p37", "p2a", "weekday(p2a)",
        "p2b", "p6", "p7", "p8", "p9",
        "p10", "p11", "p12", "p13a", "p13b",
        "p13c", "p14", "p15", "p16", "p17",
        "p18", "p19", "p20", "p21", "p22",
        "p23", "p24", "p27", "p28", "p34",
        "p35", "p39", "p44", "p45a", "p47",
        "p48a", "p49", "p50a", "p50b", "p51",
        "p52", "p53", "p55a", "p57", "p58",
        "a", "b", "d", "e", "f",
        "g", "h", "i", "j", "k",
        "l", "n", "o", "p", "q",
        "r", "s", "t", "p5a"
    ]

    types = {
        'p1': np.str_,
        'p36': np.int8,
        'p37': np.int8,
        'p2a': np.datetime64,
        'weekday(p2a)': np.int8,
        'p2b': np.str_,
        'p6': np.int8,
        'p7': np.int8,
        'p8': np.int8,
        'p9': np.int8,
        'p10': np.int8,
        'p11': np.int8,
        'p12': np.int16,
        'p13a': np.int8,
        'p13b': np.int8,
        'p13c': np.int8,
        'p14': np.int16,
        'p15': np.int8,
        'p16': np.int8,
        'p17': np.int32,
        'p18': np.int8,
        'p19': np.int8,
        'p20': np.int8,
        'p21': np.int8,
        'p22': np.int8,
        'p23': np.int8,
        'p24': np.int8,
        'p27': np.int8,
        'p28': np.int8,
        'p34': np.int8,
        'p35': np.int8,
        'p39': np.int8,
        'p44': np.int8,
        'p45a': np.int8,
        'p47': np.str_,
        'p48a': np.int8,
        'p49': np.int8,
        'p50a': np.int8,
        'p50b': np.int8,
        'p51': np.int8,
        'p52': np.int8,
        'p53': np.int16,
        'p55a': np.int8,
        'p57': np.int8,
        'p58': np.int8,
        'a': np.float64,
        'b': np.float64,
        'd': np.float64,
        'e': np.float64,
        'f': np.float64,
        'g': np.float64,
        'h': np.str_,
        'i': np.str_,
        'j': np.str_,
        'k': np.str_,
        'l': np.str_,
        'n': np.int32,
        'o': np.str_,
        'p': np.str_,
        'q': np.str_,
        'r': np.int32,
        's': np.int32,
        't': np.str_,
        'p5a': np.int8,
    }

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    def __init__(self,
                 url="https://ehw.fit.vutbr.cz/izv/",
                 folder="data",
                 cache_filename="data_{}.pkl.gz"):
        """Initialize the DataDownloader class instance."""
        self.url = url
        self.folder = folder
        self.cache_filename = cache_filename
        self.zipfiles = self.download_data(getZipfiles=True)
        self.data = {}

    def download_data(self, getZipfiles=False):
        """
        Handle website scraping and data download.

        Scrape the given (particular) webiste and
        downloads zips to be processed.
        """
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, features="lxml")

        files = []
        for line in soup.find_all('tr'):
            buttons = line.find_all('button')
            files.append(buttons[-1]['onclick'][10:-2])

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        zipfiles = []
        for file in files:
            zipfiles.append(os.path.basename(file))
            filename = f"{self.folder}/{os.path.basename(file)}"
            if os.path.isfile(filename):
                continue
            with open(filename, "wb") as f:
                f.write(requests.get(f"{self.url}/{file}").content)
        if getZipfiles:
            return zipfiles

    def purify_and_store(self, row: list, result: dict, region: str):
        """
        Clean and store data for 1 csv row.

        For a given line of csv file, this function checks
        for data inconsistencies, and sorts converted numpy
        values into their respective arrays in data dictionary.
        """
        for i, row_value in enumerate(row):
            datatype = self.types[self.headers[i]]

            default_values = {
                np.int8: -1,
                np.int16: -1,
                np.int32: -1,
                np.float64: np.nan,
            }
            if datatype == np.float64:
                row_value = row_value.replace(',', '.')
            try:
                value = datatype(row_value)
            except ValueError:
                value = default_values[datatype]
            result[self.headers[i]].append(value)

        result["region"].append(region)

    def parse_region_data(self, region: str) -> dict:
        """
        For a given region, parse csv files stored in archives.

        For a given region (three character string),
        this function downloads data from the webpage (if needed),
        unzips the archives and processes given csv files inside.
        The result will be a dictionary referenced by field/headers
        identifiers, each of which will contain numpy data
        for a given header in the particular region.
        """
        if region not in self.regions:
            raise KeyError(f"unknown region {self.region}")

        if not os.path.isdir(self.folder) or len(self.zipfiles) == 0:
            self.download_data()

        idDict = set()  # for checking duplicates

        csvRecords = {i: [] for i in self.headers}
        csvRecords['region'] = []
        for archive in self.zipfiles:
            csvName = self.regions[region] + ".csv"
            with zipfile.ZipFile(f"{self.folder}/{archive}") as zf:
                with zf.open(csvName, 'r') as csvFile:
                    reader = csv.reader(
                        io.TextIOWrapper(csvFile, encoding="cp1250"),
                        delimiter=';')
                    for row in reader:
                        if row[0] in idDict:
                            continue
                        idDict.add(row[0])
                        self.purify_and_store(row, csvRecords, region)

        csvResults = {i: np.array(csvRecords[i],
                      dtype=self.types[i]) for i in self.headers}
        csvResults['region'] = np.array(csvRecords['region'], dtype=np.str_)

        return csvResults

    def load_region_list(self, regions) -> list:
        """
        Get list of regions for parsing.

        Helper function for get_dict. Creates an array
        of region identifiers for get_dict() to process.
        """
        result = []
        if regions is None or regions == []:
            result.extend(self.regions)
        else:
            for i in regions:
                if i not in self.regions:
                    raise KeyError(f"Unknown region '{i}'")
            result.extend(regions)
        return result

    def get_dict(self, regions=None) -> dict:
        """
        Create dictionary of numpy arrays with csv values.

        Creates a dictionary of numpy arrays with concatenated
        data across all given regions. Processed data are also
        stored in a class variable, and are also stored locally
        in a cache file (location and name specified during __init__).
        """
        self.data = {i: None for i in self.regions}
        toDoRegions = self.load_region_list(regions)
        result = {i: np.empty(0, dtype=self.types[i]) for i in self.headers}
        result['region'] = np.empty(0, dtype=np.str_)
        for region in toDoRegions:
            # regionDict = None
            path = f"{self.folder}/{self.cache_filename.format(region)}"
            if self.data[region] is not None:
                regionDict = self.data[region]
            elif os.path.isfile(path):
                with gzip.open(path, 'rb', compresslevel=7) as gz:
                    regionDict = pickle.load(gz)
                    self.data[region] = regionDict
            else:
                regionDict = self.parse_region_data(region)
                with gzip.open(path, 'wb', compresslevel=7) as gz:
                    pickle.dump(regionDict, gz)
                self.data[region] = regionDict
            for header in result:
                result[header] = np.concatenate([result[header],
                                                regionDict[header]])
            result['region'] = np.concatenate([result['region'],
                                              regionDict['region']])

        return result


# TODO vypsat zakladni informace pri spusteni python3 download.py
# (ne pri importu modulu)
def main():
    """Initialize DataDownloader() and print data for 3 regions."""
    data = DataDownloader()
    myRegions = ["HKK", "JHC", "JHM"]
    result = data.get_dict(myRegions)
    for i, j in result.items():
        print(f"{i} [{j.dtype}] : [{j[0]}, {j[1]}, {j[2]}, ..., {j[-1]}]")


if __name__ == '__main__':
    main()

# End of file download.py
