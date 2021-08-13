# -*- coding: utf-8 -*-

import sys

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .common import collate_fn
from .params import TEST_BATCH_SIZE

sys.path.append("../")


def train_gcmc(
    data_loader,
    adv_loader,
    args,
    model,
    optimizer,
    optimizer_adv,
    pretrain=False,
    progress=None,
    task=None,
):
    """Training routine for GCMC, Graph Convolutional Matrix Completion
    (https://arxiv.org/abs/1706.02263v2)

    :param data_loader: Data loader for main task
    :type data_loader: torch.utils.data.DataLoader
    :param adv_loader: _unused
    :type adv_loader: None
    :param args: namespace for input parameters
    :type args: argparse.Namespace
    :param model: task model
    :type model: torch.nn.Module
    :param optimizer: optimizer for task model
    :type optimizer: torch.optim.Optimizer
    :param optimizer_adv: _unused
    :type optimizer_adv: None
    :param pretrain: _unused
    :type pretrain: bool, optional
    :param progress: _unused
    :type progress: None, optional
    :param task: _unused
    :type task: None, optional
    """

    data_itr = enumerate(data_loader)

    for idx, p_batch in data_itr:

        p_batch = p_batch.to(args.device)

        p_batch_var = p_batch

        task_loss, preds = model(p_batch_var)
        optimizer.zero_grad()
        full_loss = task_loss
        full_loss.backward(retain_graph=False)
        optimizer.step()


def test_gcmc(dataset, args, model, progress=None, task=None):
    """Testing routine for GCMC, Graph Convolutional Matrix Completion
    (https://arxiv.org/abs/1706.02263v2)

    :param dataset: Testing dataset for GCMC
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

    data_itr = enumerate(test_loader)

    preds_list = []
    rels_list = []
    test_loss_list = []
    for idx, p_batch in data_itr:
        p_batch_var = (p_batch).to(args.device)
        lhs, rel, rhs = p_batch_var[:, 0], p_batch_var[:, 1], p_batch_var[:, 2]
        test_loss, preds = model(p_batch_var)
        rel += 1
        preds_list.append(preds.squeeze())
        rels_list.append(rel.float())
        test_loss_list.append(test_loss)
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)
    test_loss = torch.mean(torch.stack(test_loss_list))

    predictions = total_preds.round().detach().cpu().numpy()
    conf = pd.DataFrame(
        confusion_matrix(total_rels.detach().cpu().numpy(), predictions)
    )

    rmse = torch.sqrt(F.mse_loss(total_preds.squeeze(), total_rels.squeeze()))

    d = {"GCMC Loss": round(test_loss.item(), 3), "RMSE": round(rmse.item(), 3)}
    return d, conf
