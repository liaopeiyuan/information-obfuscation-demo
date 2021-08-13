# -*- coding: utf-8 -*-

import itertools
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.append("../")

ltensor = torch.LongTensor

from .common import collate_fn
from .params import TEST_BATCH_SIZE


def train_gda(
    data_loader,
    adv_loader,
    args,
    model,
    optimizer_task,
    optimizer_adv,
    pretrain=False,
    progress=None,
    task=None,
):
    """Training routine for Gradient Descent-Ascent, main algorithm proposed in our work.

    :param data_loader: Data loader for main task
    :type data_loader: torch.utils.data.DataLoader
    :param adv_loader: Data loader for adversarial defense
    :type adv_loader: torch.utils.data.DataLoader
    :param args: namespace for input parameters
    :type args: argparse.Namespace
    :param model: task model
    :type model: torch.nn.Module
    :param optimizer: optimizer for task, having access to GNN and task decoder
    :type optimizer: torch.optim.Optimizer
    :param optimizer_adv: optimizer for gradient descent, having access to GNN, Gradient Reversal Layer and simulated oracle adversary
    :type optimizer_adv: torch.optim.Optimizer
    :param pretrain: whether to pretrain task without the defense pass
    :type pretrain: bool, optional
    :param progress: _unused
    :type progress: None, optional
    :param task: _unused
    :type task: None, optional
    """
    model.train()
    adv_loader = itertools.cycle(adv_loader)

    data_itr = enumerate(zip(data_loader, adv_loader))

    for idx, (p_batch, (user, gender, occupation, age)) in data_itr:

        p_batch = p_batch.to(args.device)
        (user, gender, occupation, age) = (
            user.to(args.device),
            gender.to(args.device),
            occupation.to(args.device),
            age.to(args.device),
        )

        loss_task, preds_task = model(p_batch)

        optimizer_task.zero_grad()
        loss_task.backward(retain_graph=True)
        optimizer_task.step()
        optimizer_task.zero_grad()

        if not (pretrain):
            loss_adv, (age_pred, gender_pred, occupation_pred) = model.forward_attr(
                (user, gender, occupation, age)
            )
            optimizer_adv.zero_grad()
            loss_adv.backward(retain_graph=True)
            optimizer_adv.step()
            optimizer_adv.zero_grad()


def test_gda(dataset, args, model, progress=None, task=None):
    """Testing routine for Gradient Descent-Ascent, main algorithm proposed in our work.

    :param dataset: Testing dataset for Gradient Descent-Ascent
    :type dataset: torch.utils.data.Dataset
    :param args: namespace for input parameters
    :type args: argparse.Namespace
    :param model: task model
    :type model: torch.nn.Module
    :param progress: _unused, defaults to None
    :type progress: None, optional
    :param task: _unused
    :type task: None, optional
    :return: tuple for testing metrics and pandas dataframe confusion matrix
    :rtype: Tuple[Dict[str, float], pd.DataFrame]
    """
    test_loader = DataLoader(
        dataset, batch_size=TEST_BATCH_SIZE, num_workers=1, collate_fn=collate_fn
    )
    cst_inds = np.arange(args.num_ent, dtype=np.int64)[:, None]

    data_itr = enumerate(test_loader)

    (user, gender, occupation, age) = dataset.user_features
    (user, gender, occupation, age) = (
        user.to(args.device),
        gender.to(args.device),
        occupation.to(args.device),
        age.to(args.device),
    )

    preds_list = []
    rels_list = []
    for idx, p_batch in data_itr:
        p_batch = (p_batch).to(args.device)
        lhs, rel, rhs = p_batch[:, 0], p_batch[:, 1], p_batch[:, 2]
        loss_task, preds = model(p_batch)
        loss_adv, (age_pred, gender_pred, occupation_pred) = model.forward_attr(
            (user, gender, occupation, age)
        )
        rel += 1
        preds_list.append(preds.squeeze())
        rels_list.append(rel.float())
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)

    predictions = total_preds.round().detach().cpu().numpy()
    conf = pd.DataFrame(
        confusion_matrix(total_rels.detach().cpu().numpy(), predictions)
    )

    rmse = torch.sqrt(F.mse_loss(total_preds.squeeze(), total_rels.squeeze()))

    d = {"Adv Loss": round(loss_adv.item(), 3), "RMSE": round(rmse.item(), 3)}
    return d, conf
