# -*- coding: utf-8 -*-

from models import (GAL_Neighbor, NeighborClassifier, NodeClassifier,
                    SharedBilinearDecoder)
from node_attack import NodeAttackRunner


class NeighborAttackRunner(NodeAttackRunner):
    def __init__(self, args):
        super().__init__(args)

    def get_base_model(self):
        decoder = SharedBilinearDecoder(
            self.args.num_rel, 2, self.args.embed_dim, self.args
        ).to(self.args.device)
        model = GAL_Neighbor(
            decoder, self.args.embed_dim, self.args.num_ent, self.edges, self.args
        ).to(self.args.device)
        return model

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
