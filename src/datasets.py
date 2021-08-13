# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

from helper import MOVIELENS_1M_DIR


class GDADataset(Dataset):
    """Dataset that collects targets for link prediction (task)
    for Gradient Descent-Ascent.

    :param edge_data: edge-level data
    :type edge_data: pd.DataFrame
    :param node_data: node-level data
    :type node_data: pd.DataFrame
    :param prefetch_to_gpu: whether to pre-load data to GPU, defaults to False
    :type prefetch_to_gpu: bool, optional
    """

    def __init__(
        self, edge_data: pd.DataFrame, node_data: pd.DataFrame, prefetch_to_gpu=False
    ):
        """Constructor method"""
        self.prefetch_to_gpu = prefetch_to_gpu
        self.edge_data = np.ascontiguousarray(edge_data)
        self.node_data = list(node_data)

        users = pd.read_csv(os.path.join(MOVIELENS_1M_DIR, "ml-1m", "users.csv"))

        label_encoder = preprocessing.LabelEncoder()

        self.gender = label_encoder.fit_transform(users.sex.values).reshape(-1, 1)
        self.occupation = users.occupation.values
        self.age = users.age.values
        self.age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

        stack = []

        for user in self.node_data:
            stack.append(
                [
                    user,
                    self.gender[user, :],
                    self.occupation[user],
                    self.age_dict[self.age[user]],
                ]
            )

        users_l = []
        genders_l = []
        occupations_l = []
        ages_l = []

        for [user, gender, occupation, age] in stack:
            users_l.append(user)
            genders_l.append(gender)
            occupations_l.append(occupation)
            ages_l.append(age)

        users = torch.LongTensor(users_l)
        genders = torch.LongTensor(genders_l)
        occupations = torch.LongTensor(occupations_l)
        ages = torch.LongTensor(ages_l)

        self.user_features = (users, genders, occupations, ages)

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return self.edge_data[idx]


class NodeClassification(Dataset):
    """Dataset that loads node classification targets, used both by the adversarial defense
    in Gradient Descent-Ascent as well as oracle adversary during testing.

    :param data: training dataset
    :type data: pd.DataFrame
    :param prefetch_to_gpu: whether to pre-load data to GPU, defaults to False
    :type prefetch_to_gpu: bool, optional
    """

    def __init__(self, data: pd.DataFrame, prefetch_to_gpu=False):
        """Constructor method"""
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data)

        users = pd.read_csv(os.path.join(MOVIELENS_1M_DIR, "ml-1m", "users.csv"))

        label_encoder = preprocessing.LabelEncoder()

        self.gender = label_encoder.fit_transform(users.sex.values).reshape(-1, 1)
        self.occupation = users.occupation.values
        self.age = users.age.values
        self.age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user = self.dataset[idx]

        return [
            user,
            self.gender[user, :],
            self.occupation[user],
            self.age_dict[self.age[user]],
        ]

    def shuffle(self):
        """Shuffle dataset"""
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        data = np.ascontiguousarray(data)
        self.dataset = torch.LongTensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()


class KBDataset(Dataset):
    """Dataset that collects targets for link prediction (task)
    for baseline embedding model.

    :param data: training data
    :type data: pd.DataFrame
    :param users: user data
    :type users: pd.DataFrame
    :param prefetch_to_gpu: whether to pre-load data to GPU, defaults to False
    :type prefetch_to_gpu: bool, optional
    """

    def __init__(self, data: pd.DataFrame, users: pd.DataFrame, prefetch_to_gpu=False):
        """Constructor method"""
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data)
        self.users = users

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        """Shuffle dataset"""
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = torch.LongTensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()
