# -*- coding: utf-8 -*-

from models import NeighborClassifier, NodeClassifier
from node_attack_gcmc import NodeAttackGCMCRunner


class NeighborAttackGCMCRunner(NodeAttackGCMCRunner):
    def __init__(self, args):
        super().__init__(args)

    def num_adversaries(self):
        return 2

    def get_ordered_adversary_names(self):
        return ["Node", "Neighbor"]

    def get_adversary_models(self, mode):
        embeddings = self.base_model.encode(None).detach().squeeze(0)
        return [
            NodeClassifier(self.args.embed_dim, embeddings).to(self.args.device),
            NeighborClassifier(self.args.embed_dim, embeddings, self.edges).to(
                self.args.device
            ),
        ]


if __name__ == "__main__":
    assert False  # You shouldn't run this. Please call exec.py
