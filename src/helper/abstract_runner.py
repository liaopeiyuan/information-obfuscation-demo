# -*- coding: utf-8 -*-

import abc
import argparse
import gc
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from rich.progress import (BarColumn, Progress, TaskID, TextColumn,
                           TimeRemainingColumn)
from torch.utils.data import DataLoader

from .common import collate_fn, node_cls_collate_fn
from .params import MOVIELENS_1M_DIR


class Runner(abc.ABC):
    """Abstract runner class for task training and attribute inference attacks

    :param args: namespace for input parameters
    :type args: argparse.Namespace
    """

    def __init__(self, args: argparse.Namespace):
        """Constructor method"""
        self.args = args
        self.prefetch_to_gpu = args.prefetch_to_gpu
        self.train_set = self.build_main_dataset(
            args.train_ratings, args.users_train, args.prefetch_to_gpu
        )
        self.test_set = self.build_main_dataset(
            args.test_ratings, args.users_test, args.prefetch_to_gpu
        )
        self.adv_train_set = self.bulid_adversary_dataset(
            args.users_train, args.prefetch_to_gpu
        )
        self.adv_test_set = self.bulid_adversary_dataset(
            args.users_test, args.prefetch_to_gpu
        )
        self.edges = self.build_edges(args.train_ratings)
        self.base_model = self.get_base_model()
        self.train_routine = self.get_train_routine()
        self.test_routine = self.get_test_routine()
        self.adv_train_routine = self.get_adv_train_routine()
        self.adv_test_routine = self.get_adv_test_routine()

        train_loader_params = dict(
            dataset=self.train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        adv_cls_loader_params = dict(
            dataset=self.adv_train_set,
            batch_size=args.node_cls_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=node_cls_collate_fn,
        )

        if args.prefetch_to_gpu:
            adv_cls_loader_params["pin_memory"] = True
            train_loader_params["pin_memory"] = True

        self.train_loader = DataLoader(**train_loader_params)
        self.adv_train_loader = DataLoader(**adv_cls_loader_params)

    @abc.abstractmethod
    def get_test_routine(
        self,
    ) -> Callable[
        [
            torch.utils.data.Dataset,
            argparse.Namespace,
            torch.nn.Module,
            Progress,
            TaskID,
        ],
        Tuple[Dict[str, float], pd.DataFrame],
    ]:
        """Returns function to perform objective task evaluation

        :return: function to perform objective task evaluation
        :rtype: Callable[ [ torch.utils.data.Dataset, argparse.Namespace, torch.nn.Module, Progress, TaskID, ], Tuple[Dict[str, float], pd.DataFrame], ]
        """
        pass

    @abc.abstractmethod
    def get_train_routine(
        self,
    ) -> Callable[
        [
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            argparse.Namespace,
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.Optimizer,
            Progress,
            TaskID,
        ],
        None,
    ]:
        """Returns function to perform objective task training

        :return: function to perform objective task training
        :rtype: Callable[ [ torch.utils.data.DataLoader, torch.utils.data.DataLoader, argparse.Namespace, torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer, Progress, TaskID, ], None, ]
        """
        pass

    @abc.abstractmethod
    def get_adv_test_routine(
        self,
    ) -> Callable[
        [
            torch.utils.data.DataLoader,
            argparse.Namespace,
            torch.nn.Module,
            torch.optim.Optimizer,
            Progress,
            TaskID,
        ],
        None,
    ]:
        """Returns function to perform adversary evaluation

        :return: function to perform adversary evaluation
        :rtype: Callable[ [ torch.utils.data.DataLoader, argparse.Namespace, torch.nn.Module, torch.optim.Optimizer, Progress, TaskID, ], None, ]
        """
        pass

    @abc.abstractmethod
    def get_adv_train_routine(
        self,
    ) -> Callable[
        [
            torch.utils.data.Dataset,
            argparse.Namespace,
            torch.nn.Module,
            str,
            Progress,
            TaskID,
        ],
        Tuple[Dict[str, float], pd.DataFrame],
    ]:
        """Returns function to perform adversary training

        :return: function to perform adversary training
        :rtype: Callable[ [ torch.utils.data.Dataset, argparse.Namespace, torch.nn.Module, str, Progress, TaskID, ], Tuple[Dict[str, float], pd.DataFrame], ]
        """
        pass

    @abc.abstractmethod
    def get_main_dataset(self) -> pd.DataFrame:
        """Returns main dataset for objective task

        :return: main dataset for objective task
        :rtype: pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def get_adv_classification_dataset(self) -> pd.DataFrame:
        """Returns dataset for adversarial task

        :return: dataset for adversarial task
        :rtype: pd.DataFrame
        """
        pass

    def build_main_dataset(
        self, ratings: pd.DataFrame, users: pd.DataFrame, prefetch_to_gpu: bool
    ) -> torch.utils.data.Dataset:
        """Routine to build PyTorch dataset from pandas dataframes for task training

        :param ratings: movie ratings
        :type ratings: pd.DataFrame
        :param users: user data
        :type users: pd.DataFrame
        :param prefetch_to_gpu: whether to pre-load data to gpu
        :type prefetch_to_gpu: bool
        :return: PyTorch dataset for main task
        :rtype: torch.utils.data.Dataset
        """
        main_dataset_builder = self.get_main_dataset()
        return main_dataset_builder(ratings, users, prefetch_to_gpu)

    def bulid_adversary_dataset(
        self, users: pd.DataFrame, prefetch_to_gpu: bool
    ) -> torch.utils.data.Dataset:
        """Routine to build PyTorch dataset from pandas dataframes for adversarial task

        :param users: user data
        :type users: pd.DataFrame
        :param prefetch_to_gpu: whether to pre-load data to gpu
        :type prefetch_to_gpu: bool
        :return: PyTorch dataset for adversarial task
        :rtype: torch.utils.data.Dataset
        """
        adv_dataset_builder = self.get_adv_classification_dataset()
        return adv_dataset_builder(users, prefetch_to_gpu)

    def build_edges(self, train_ratings: pd.DataFrame) -> torch.LongTensor:
        """Routine to build edges for graph neural networks

        :param train_ratings: train movie ratings
        :type train_ratings: pd.DataFrame
        :return: edge tensor
        :rtype: torch.LongTensor
        """
        edges = np.hstack(
            (
                np.stack(
                    [
                        train_ratings["user_id"].values,
                        train_ratings["movie_id"].values,
                    ]
                ),
                np.stack(
                    [
                        train_ratings["movie_id"].values,
                        train_ratings["user_id"].values,
                    ]
                ),
            )
        )
        edges = torch.LongTensor(edges)
        return edges

    @abc.abstractmethod
    def get_base_model(self) -> torch.nn.Module:
        """Returns the base model for main task and optionally adversarial defense branch

        :return: base model
        :rtype: torch.nn.Module
        """
        pass

    @abc.abstractmethod
    def get_adversary_models(self, mode: str) -> List[torch.nn.Module]:
        """Returns a list of adversaries attacking different parts of the network

        :param mode: sensitive attribute to attack (one of age, gender or occupation)
        :type mode: str
        :return: list of adversaries
        :rtype: List[torch.nn.Module]
        """
        pass

    @abc.abstractmethod
    def get_oracle_adversary_optimizers(
        self, adversaries: List[torch.nn.Module], mode: str
    ) -> List[torch.optim.Optimizer]:
        """Returns a list of optimizers corresponding to the adversaries

        :param adversaries: list of adversaries attacking different parts of the network
        :type adversaries: List[torch.nn.Module]
        :param mode: sensitive attribute to attack (one of age, gender or occupation)
        :type mode: str
        :return: list of optimizers
        :rtype: List[torch.optim.Optimizer]
        """
        pass

    @abc.abstractmethod
    def get_task_optimizer(self) -> torch.optim.Optimizer:
        """Returns the optimizer corresponding to the task model

        :return: optimizer corresponding to the task model
        :rtype: torch.optim.Optimizer
        """
        pass

    @abc.abstractmethod
    def get_adv_optimizer(self, mode: str) -> torch.optim.Optimizer:
        """Returns the optimizer corresponding to the adversary model

        :param mode: sensitive attribute to attack (one of age, gender or occupation)
        :type mode: str
        :return: optimizer corresponding to the adversary model
        :rtype: torch.optim.Optimizer
        """
        pass

    @abc.abstractmethod
    def num_adversaries(self) -> int:
        """Returns the numebr of adversaries

        :return: numebr of adversaries
        :rtype: int
        """
        pass

    @abc.abstractmethod
    def get_ordered_adversary_names(self) -> List[str]:
        """Returns a list of adversary names, in the same order as the model and optimizer method

        :return: list of adversary names
        :rtype: List[str]
        """
        pass

    def train_task_with_adversary(
        self,
        mode: str,
        dirname: str,
        refresh=True,
        progress=None,
        task=None,
        adv_tasks=[],
    ):
        """Train a task, then evaluate adversary performance

        :param mode: sensitive attribute to evaluate (one of age, occupation and gender)
        :type mode: str
        :param dirname: directory to store model checkpoints
        :type dirname: str
        :param refresh: whether to re-initialize model for each sensitive attribute, defaults to True
        :type refresh: bool, optional
        :param progress: progress bar instance, defaults to None
        :type progress: rich.progress.Progress, optional
        :param task: task instance, defaults to None
        :type task: rich.progress.TaskID, optional
        :param adv_tasks: list of adversaries to test, defaults to []
        :type adv_tasks: list, optional
        """

        if refresh:
            self.base_model = self.get_base_model()

        optimizer_task = self.get_task_optimizer()
        optimizer_adv = self.get_adv_optimizer(mode)

        self.base_model.set_mode(mode)

        for epoch in range(self.args.num_epochs):

            if epoch % (self.args.valid_freq) == 0:
                with torch.no_grad():
                    task_measures, task_cm = self.test_routine(
                        self.test_set, self.args, self.base_model, progress, task
                    )

            self.train_routine(
                self.train_loader,
                self.adv_train_loader,
                self.args,
                self.base_model,
                optimizer_task,
                optimizer_adv,
                False,
            )
            gc.collect()

            wandb.log(task_measures)
            task_tb = wandb.Table(
                columns=list(task_cm.columns), data=task_cm.values.tolist()
            )
            wandb.log({"Task Confusion Matrix": task_tb})
            self.args.logger.info(
                [epoch, "task", str(task_measures), str(task_cm.to_dict())]
            )
            progress.update(
                task,
                advance=1,
                measures=task_measures,
                conf_matrix=task_cm,
                refresh=True,
            )

        path = os.path.join(dirname, "model.pth")
        torch.save(self.base_model.state_dict(), path)

        oracle_adversaries = self.get_adversary_models(mode)
        optimizer_oracle_adversaries = self.get_oracle_adversary_optimizers(
            oracle_adversaries, mode
        )

        for name, oracle_adversary, optimizer_oracle_adversary, adv_task in zip(
            self.get_ordered_adversary_names(),
            oracle_adversaries,
            optimizer_oracle_adversaries,
            adv_tasks,
        ):
            oracle_adversary.set_mode(mode)

            for epoch in range(self.args.finetune_epochs):
                self.adv_train_routine(
                    self.adv_train_loader,
                    self.args,
                    oracle_adversary,
                    optimizer_oracle_adversary,
                    progress,
                    task,
                )
                gc.collect()
                with torch.no_grad():
                    adversary_measures, adversary_cm = self.adv_test_routine(
                        self.adv_test_set,
                        self.args,
                        oracle_adversary,
                        mode,
                        progress,
                        task,
                    )

                wandb.log(adversary_measures)
                adversary_tb = wandb.Table(
                    columns=list(adversary_cm.columns),
                    data=adversary_cm.values.tolist(),
                )
                wandb.log({"Adversary Confusion Matrix": adversary_tb})
                self.args.logger.info(
                    [
                        epoch,
                        "adversary",
                        str(adversary_measures),
                        str(adversary_cm.to_dict()),
                    ]
                )
                progress.update(
                    adv_task,
                    advance=1,
                    measures=adversary_measures,
                    conf_matrix=adversary_cm,
                    refresh=True,
                )

            path = os.path.join(dirname, f"adversary_{mode}_{name}.pth")
            torch.save(oracle_adversary.state_dict(), path)

    def run(self, refresh=True):
        """Running task training, then evaluate adversaries in three sensitive attributes (gender, age, occupation)

        :param refresh: whether to re-initialize model for each sensitive attribute, defaults to True
        :type refresh: bool, optional
        """

        progress = Progress(
            "[progress.description]{task.description}",
            TextColumn("[bold green]{task.fields[measures]}", justify="right"),
            TextColumn(
                "[dark_goldenrod]Truncated CM {task.fields[conf_matrix]}",
                justify="right",
            ),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            auto_refresh=False,
        )

        logname = self.args.logname
        print("Log stored at: ", logname)
        run = wandb.init(
            project="information-obfuscation",
            entity="peiyuanl",
            name=logname,
            config=vars(self.args),
        )
        dirname = os.path.join(
            "../checkpoints",
            self.args.experiment,
            self.args.task,
            self.args.model,
            logname,
        )
        Path(dirname).mkdir(parents=True, exist_ok=True)

        with progress:
            gender_adv_tasks = []
            age_adv_tasks = []
            occupation_adv_tasks = []

            # To ensure layout correctness

            gender_task = progress.add_task(
                "[cyan]Gender Task",
                total=self.args.num_epochs,
                measures={},
                conf_matrix=[],
            )
            for name in self.get_ordered_adversary_names():
                gender_adv_tasks.append(
                    progress.add_task(
                        f"[cyan]Gender {name} Adversary",
                        total=self.args.finetune_epochs,
                        measures={},
                        conf_matrix=[],
                    )
                )

            age_task = progress.add_task(
                "[cyan]Age Task",
                total=self.args.num_epochs,
                measures={},
                conf_matrix=[],
            )
            for name in self.get_ordered_adversary_names():
                age_adv_tasks.append(
                    progress.add_task(
                        f"[cyan]Age {name} Adversary",
                        total=self.args.finetune_epochs,
                        measures={},
                        conf_matrix=[],
                    )
                )

            occupation_task = progress.add_task(
                "[cyan]Occupation Task",
                total=self.args.num_epochs,
                measures={},
                conf_matrix=[],
            )

            for name in self.get_ordered_adversary_names():
                occupation_adv_tasks.append(
                    progress.add_task(
                        f"[cyan]Age {name} Adversary",
                        total=self.args.finetune_epochs,
                        measures={},
                        conf_matrix=[],
                    )
                )

            self.train_task_with_adversary(
                "gender",
                dirname,
                refresh=refresh,
                progress=progress,
                task=gender_task,
                adv_tasks=gender_adv_tasks,
            )
            self.train_task_with_adversary(
                "age",
                dirname,
                refresh=refresh,
                progress=progress,
                task=age_task,
                adv_tasks=age_adv_tasks,
            )
            self.train_task_with_adversary(
                "occupation",
                dirname,
                refresh=refresh,
                progress=progress,
                task=occupation_task,
                adv_tasks=occupation_adv_tasks,
            )

        trained_model_artifact = wandb.Artifact(
            logname + "_model", type="model", description="Task and adversary models"
        )
        trained_model_artifact.add_dir(dirname)
        run.log_artifact(trained_model_artifact)

        dataset_artifact = wandb.Artifact(
            logname + "_dataset",
            type="dataset",
            description="Dataset used to train the models",
        )
        dataset_artifact.add_dir(MOVIELENS_1M_DIR)
        run.log_artifact(dataset_artifact)
