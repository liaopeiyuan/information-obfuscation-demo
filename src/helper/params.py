# -*- coding: utf-8 -*-

import pathlib
import warnings
from typing import List

#                                  helper src    project
dirs = list(pathlib.Path(__file__).parent.parent.parent.resolve().glob("data"))

assert (
    dirs
), "directory data does not exist in project. Please download Movielens data at https://grouplens.org/datasets/movielens/1m/"

MOVIELENS_1M_DIR = dirs[0]
IS_DATA_PROCESSED = True
MAX_NUM_USERS = 9992
TEST_BATCH_SIZE = 4000
TEST_NAIVE_CLS_THRESHOLD = 0.5
TRAIN_SPLIT = 0.9
OCCUPATION_LIST = [
    "other",
    "adaemic",
    "artist",
    "cleri.",
    "grad st.",
    "custom.",
    "doctor",
    "exec.",
    "farmer",
    "homema.",
    "K-12 st.",
    "lawyer",
    "prog.",
    "retired",
    "sales",
    "scient.",
    "self-em",
    "techn.",
    "trades.",
    "unemp.",
    "writer",
]
GENDER_LIST = ["F", "M"]
AGE_LIST = ["18-", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
NUM_WEIGHTS = 2

csv_files = list(MOVIELENS_1M_DIR.glob("ml-1m/*.csv"))
dat_files = list(MOVIELENS_1M_DIR.glob("ml-1m/*.dat"))

dats = ["ratings.dat", "users.dat", "movies.dat"]
csvs = ["ratings.csv", "users.csv", "movies.csv"]


def filter_file_existence(c: List[str], files: List[pathlib.Path]) -> List[str]:
    """Filter for files that are not found

    :param c: list of files to find
    :type c: List[str]
    :param files: list of file paths
    :type files: List[pathlib.Path]
    :return: list of file names that are not found
    :rtype: List[str]
    """
    for x in files:
        for cand in c:
            if cand in str(x):
                c.remove(cand)
                break
    return c


dats = filter_file_existence(dats, dat_files)
csvs = filter_file_existence(csvs, csv_files)

assert (
    not dats
), f"{dats} not found under directory {MOVIELENS_1M_DIR}/ml-1m . Please run download.sh"
if csvs:
    IS_DATA_PROCESSED = False
    warnings.warn(
        "{} not found under directory {}/ml-1m . Will call make_dataset_1M before running.".format(
            csvs, MOVIELENS_1M_DIR
        )
    )
