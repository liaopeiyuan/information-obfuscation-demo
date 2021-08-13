# -*- coding: utf-8 -*-

import sys

import numpy as np
import torch

sys.path.append("../")

to_np = lambda x: x.detach().cpu().numpy()


def to_device(tensor):
    if tensor is not None:
        return tensor.to("cuda")


ltensor = torch.LongTensor


def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (
        isinstance(batch, list) and isinstance(batch[0], np.ndarray)
    ):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()


def node_cls_collate_fn(batch):
    users = []
    genders = []
    occupations = []
    ages = []
    for [user, gender, occupation, age] in batch:
        users.append(user)
        genders.append(gender)
        occupations.append(occupation)
        ages.append(age)

    users = ltensor(users)
    genders = ltensor(genders)
    occupations = ltensor(occupations)
    ages = ltensor(ages)

    return (users, genders, occupations, ages)
