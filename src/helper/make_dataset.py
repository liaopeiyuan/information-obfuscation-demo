# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from .params import MOVIELENS_1M_DIR, TRAIN_SPLIT


def make_dataset_1M(
    processed=False, seed=42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads/Builds the Movielens-1M dataset

    :param processed: if the dataset is already built. If so, we will just load them from the local directory, defaults to False
    :type processed: bool, optional
    :param seed: random seed for train/validation split, defaults to 42
    :type seed: int, optional
    :return: pandas dataframes (train_ratings, test_ratings, users, movies) for training, validation and test
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    ml_1m_path = os.path.join(MOVIELENS_1M_DIR, "ml-1m/")
    if processed:
        csvs_to_load = [
            "train_ratings.csv",
            "test_ratings.csv",
            "users.csv",
            "movies.csv",
        ]
        [train_ratings, test_ratings, users, movies] = map(
            lambda x: pd.read_csv(os.path.join(MOVIELENS_1M_DIR, "ml-1m/", x)),
            csvs_to_load,
        )
        return train_ratings, test_ratings, users, movies
    else:
        r_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]

        ratings = pd.read_csv(
            os.path.join(ml_1m_path, "ratings.dat"),
            sep="::",
            names=r_cols,
            encoding="latin-1",
            engine="python",
        )
        shuffled_ratings = ratings.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )  # Shuffle

        train_cutoff_row = int(np.round(len(shuffled_ratings) * TRAIN_SPLIT))
        train_ratings = shuffled_ratings[:train_cutoff_row]
        test_ratings = shuffled_ratings[train_cutoff_row:]

        u_cols = ["user_id", "sex", "age", "occupation", "zip_code"]
        m_cols = ["movie_id", "title", "genre"]
        users = pd.read_csv(
            os.path.join(ml_1m_path, "users.dat"),
            sep="::",
            names=u_cols,
            encoding="latin-1",
            parse_dates=True,
            engine="python",
        )
        movies = pd.read_csv(
            os.path.join(ml_1m_path, "movies.dat"),
            sep="::",
            names=m_cols,
            encoding="latin-1",
            parse_dates=True,
            engine="python",
        )

        train_ratings.drop("unix_timestamp", inplace=True, axis=1)
        train_ratings_matrix = train_ratings.pivot_table(
            index=["movie_id"], columns=["user_id"], values="rating"
        ).reset_index(drop=True)
        test_ratings.drop("unix_timestamp", inplace=True, axis=1)
        columnsTitles = ["user_id", "rating", "movie_id"]
        train_ratings = train_ratings.reindex(columns=columnsTitles) - 1
        test_ratings = test_ratings.reindex(columns=columnsTitles) - 1
        users.user_id = users.user_id.astype(np.int64)
        movies.movie_id = movies.movie_id.astype(np.int64)
        users["user_id"] = users["user_id"] - 1
        movies["movie_id"] = movies["movie_id"] - 1

        """ Offset Movie ID's by # users because in TransD they share the same
        embedding Layer """

        train_ratings["movie_id"] += int(np.max(users["user_id"])) + 1
        test_ratings["movie_id"] += int(np.max(users["user_id"])) + 1

        users.to_csv(os.path.join(ml_1m_path, "users.csv"), index=False)
        movies.to_csv(os.path.join(ml_1m_path, "movies.csv"), index=False)
        train_ratings.to_csv(os.path.join(ml_1m_path, "train_ratings.csv"), index=False)
        test_ratings.to_csv(os.path.join(ml_1m_path, "test_ratings.csv"), index=False)

        return train_ratings, test_ratings, users, movies
