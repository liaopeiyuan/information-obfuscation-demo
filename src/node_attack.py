# -*- coding: utf-8 -*-

from datasets import GDADataset, NodeClassification
from helper import (Runner, create_optimizer, test_gda, test_node_cls,
                    train_gda, train_node_cls)
from models import GAL, NodeClassifier, SharedBilinearDecoder


class NodeAttackRunner(Runner):
    def __init__(self, args):
        super().__init__(args)

    def get_test_routine(self):
        return test_gda

    def get_train_routine(self):
        return train_gda

    def get_adv_test_routine(self):
        return test_node_cls

    def get_adv_train_routine(self):
        return train_node_cls

    def get_main_dataset(self):
        return GDADataset

    def get_adv_classification_dataset(self):
        return NodeClassification

    def get_base_model(self):
        decoder = SharedBilinearDecoder(
            self.args.num_rel, 2, self.args.embed_dim, self.args
        ).to(self.args.device)
        model = GAL(
            decoder, self.args.embed_dim, self.args.num_ent, self.edges, self.args
        ).to(self.args.device)
        return model

    def num_adversaries(self):
        return 1

    def get_ordered_adversary_names(self):
        return ["Node"]

    def get_adversary_models(self, mode):
        embeddings = self.base_model.encode(None).detach().squeeze(0)
        return [
            NodeClassifier(self.args.embed_dim, embeddings).to(self.args.device),
        ]

    def get_oracle_adversary_optimizers(self, adversaries, mode):
        ret = []

        for adversary in adversaries:
            if mode == "gender":
                task_specific_params = adversary.gender.parameters()
            elif mode == "age":
                task_specific_params = adversary.age.parameters()
            else:
                assert mode == "occupation"
                task_specific_params = adversary.occupation.parameters()
            ret.append(create_optimizer(task_specific_params, "adam", self.args.lr))
        return ret

    def get_task_optimizer(self):
        optimizer_task = create_optimizer(
            [
                {"params": self.base_model.encoder.parameters()},
                {"params": self.base_model.batchnorm.parameters()},
                {"params": self.base_model.decoder.parameters()},
                {"params": self.base_model.gnn.parameters()},
            ],
            "adam",
            self.args.lr,
        )
        return optimizer_task

    def get_adv_optimizer(self, mode):
        if mode == "gender":
            task_specific_params = {"params": self.base_model.gender.parameters()}
        elif mode == "age":
            task_specific_params = {"params": self.base_model.age.parameters()}
        else:
            assert mode == "occupation"
            task_specific_params = {"params": self.base_model.occupation.parameters()}

        optimizer_adv = create_optimizer(
            [
                {"params": self.base_model.encoder.parameters()},
                {"params": self.base_model.batchnorm.parameters()},
                task_specific_params,
                {"params": self.base_model.gnn.parameters()},
            ],
            "adam",
            self.args.lr * self.args.lambda_reg,
        )
        return optimizer_adv


if __name__ == "__main__":
    assert False  # You shouldn't run this. Please call exec.py
