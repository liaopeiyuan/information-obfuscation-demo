# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .common import node_cls_collate_fn, to_np
from .params import (AGE_LIST, GENDER_LIST, OCCUPATION_LIST, TEST_BATCH_SIZE,
                     TEST_NAIVE_CLS_THRESHOLD)

sys.path.append("../")


def train_node_cls(data_loader, args, model, optimizer, progress=None, task=None):
    """Training routine for node classification on graphs.

    :param data_loader: Data loader for main task
    :type data_loader: torch.utils.data.DataLoader
    :param args: namespace for input parameters
    :type args: argparse.Namespace
    :param model: node classification model
    :type model: torch.nn.Module
    :param optimizer: optimizer for node classification model
    :type optimizer: torch.optim.Optimizer
    :param progress: _unused
    :type progress: None, optional
    :param task: _unused
    :type task: None, optional
    """
    model.train()

    data_itr = enumerate(data_loader)

    for idx, (user, gender, occupation, age) in data_itr:

        (user, gender, occupation, age) = (
            user.to(args.device),
            gender.to(args.device),
            occupation.to(args.device),
            age.to(args.device),
        )

        task_loss, preds = model((user, gender, occupation, age))
        optimizer.zero_grad()
        full_loss = task_loss
        full_loss.backward()

        optimizer.step()


def test_node_cls(test_node_cls_set, args, model, mode="age", progress=None, task=None):
    """Testing routine for node classification on graphs.

    :param test_node_cls_set: Testing dataset for node classification
    :type test_node_cls_set: torch.utils.data.Dataset
    :param args: namespace for input parameters
    :type args: argparse.Namespace
    :param model: task model
    :type model: torch.nn.Module
    :param mode: attribute to classifiy (one of age, gender or occupation), defaults to "age"
    :type mode: str, optional
    :param progress: _unused, defaults to None
    :type progress: None, optional
    :param task: _unused
    :type task: None, optional
    :return: tuple for testing metrics and pandas dataframe confusion matrix
    :rtype: Tuple[Dict[str, float], pd.DataFrame]
    """
    model.eval()
    node_cls_test_loader = DataLoader(
        test_node_cls_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=node_cls_collate_fn,
    )

    predictions = []
    truths = []

    for idx, (user, gender, occupation, age) in enumerate(node_cls_test_loader):
        (user, gender, occupation, age) = (
            user.to(args.device),
            gender.to(args.device),
            occupation.to(args.device),
            age.to(args.device),
        )

        task_loss, [pred_age, pred_gender, pred_occupation] = model(
            (user, gender, occupation, age)
        )

        pred_age = pred_age.max(1)[1]
        pred_occupation = pred_occupation.max(1)[1]
        pred_gender = pred_gender > TEST_NAIVE_CLS_THRESHOLD

        pred_age, truth_age = to_np(pred_age), to_np(age)
        pred_occupation, truth_occupation = to_np(pred_occupation), to_np(occupation)
        pred_gender, truth_gender = to_np(pred_gender), to_np(gender)

        if mode == "gender":
            predictions.append(pred_gender)
            truths.append(truth_gender)
        elif mode == "age":
            predictions.append(pred_age)
            truths.append(truth_age)
        elif mode == "occupation":
            predictions.append(pred_occupation)
            truths.append(truth_occupation)

    truth = np.concatenate(truths, axis=0)
    prediction = np.concatenate(predictions, axis=0)

    if mode == "gender":
        d = {"AUC": round(roc_auc_score(truth_gender, pred_gender), 3)}
    else:
        d = {"Macro F1": round(f1_score(prediction, truth, average="macro"), 3)}

    if mode == "gender":
        names = GENDER_LIST
    elif mode == "age":
        names = AGE_LIST
    elif mode == "occupation":
        names = OCCUPATION_LIST

    conf = confusion_matrix(truth, prediction)
    conf = pd.DataFrame(conf)
    conf.columns = names
    conf.index = names
    conf = conf.iloc[:, :6].head(6)

    return d, conf
