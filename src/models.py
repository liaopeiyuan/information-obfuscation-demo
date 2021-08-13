# -*- coding: utf-8 -*-

import argparse
import math
import warnings
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, Linear, Sequential
from torch_geometric.nn import ChebConv, GATConv, GCNConv, GINConv, SAGEConv

from helper import MAX_NUM_USERS, ltensor, to_device


class GNN(torch.nn.Module):
    """Multi-layer Graph Neural Network supporting various neighborhood aggregation methods

    :param embed: embedding dimension
    :type embed: int
    :param gnn_layers: number of GNN layers
    :type gnn_layers: int
    :param gnn_type: type of GNN
    :type gnn_type: str
    :param device: deivce to run the neural network
    :type device: Union[str, torch.device]
    :raises NotImplementedError: GNN type not recognized
    """

    def __init__(
        self,
        embed: int,
        gnn_layers: int,
        gnn_type: str,
        device: Union[str, torch.device],
    ):
        """Constructor method"""
        super(GNN, self).__init__()
        h = embed

        def get_layer(gnn_type):
            if gnn_type == "ChebConv":
                layer = ChebConv(h, h, K=2)
            elif gnn_type == "GCNConv":
                layer = GCNConv(h, h)
            elif gnn_type == "GINConv":
                dnn = Sequential(Linear(h, h), LeakyReLU(), Linear(h, h))
                layer = GINConv(dnn)
            elif gnn_type == "SAGEConv":
                layer = SAGEConv(h, h, normalize=True)
            elif gnn_type == "GATConv":
                layer = GATConv(h, h)
            else:
                raise NotImplementedError
            return layer

        self.convs = []
        for _ in range(gnn_layers):
            self.convs.append(get_layer(gnn_type).to(device))

    def forward(
        self, embeddings: torch.FloatTensor, edge_index: torch.LongTensor
    ) -> torch.FloatTensor:
        """Generate graph-neural encoded embeddings

        :param embeddings: learned embeddings
        :type embeddings: torch.FloatTensor
        :param edge_index: edge index tensor
        :type edge_index: torch.LongTensor
        :return: graph-neural encoded embeddings
        :rtype: torch.FloatTensor
        """

        for layer in self.convs:
            embeddings = layer(embeddings, edge_index)

        return embeddings


class GradReverse(torch.autograd.Function):
    """Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, inputs):
        return GradReverse.apply(inputs)


class SharedBilinearDecoder(nn.Module):
    """Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    Modified from https://github.com/joeybose/Flexible-Fairness-Constraints

    :param num_relations: number of relations to predict
    :type num_relations: int
    :param num_weights: number of base weights
    :type num_weights: int
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param args: namespace for input parameters
    :type args: argparse.Namespace

    """

    def __init__(
        self,
        num_relations: int,
        num_weights: int,
        embed_dim: int,
        args: argparse.Namespace,
    ):
        """Constructor method"""
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim * embed_dim)
        self.weight_scalars = nn.Parameter(torch.Tensor(num_weights, num_relations))
        stdv = 1.0 / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()
        self.args = args

    def predict(
        self, embeds1: torch.FloatTensor, embeds2: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Predict scoring by weighted sum of score probabilities and score.

        :param embeds1: the first embedding in biliear form
        :type embeds1: torch.FloatTensor
        :param embeds2: the second embedding in biliear form
        :type embeds2: torch.FloatTensor
        :return: predicted scores
        :rtype: torch.FloatTensor
        """
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (ltensor([i])).to(self.args.device)
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        preds = 0
        for j in range(0, self.num_relations):
            index = (ltensor([j])).to(self.args.device)
            """ j+1 because of zero index """
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return preds

    def forward(
        self,
        embeds1: torch.FloatTensor,
        embeds2: torch.FloatTensor,
        rels: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Predict scoring by weighted sum of score probabilities and score as well as computing the NLL loss.

        :param embeds1: the first embedding in biliear form
        :type embeds1: torch.FloatTensor
        :param embeds2: the second embedding in biliear form
        :type embeds2: torch.FloatTensor
        :param rels: All relationships
        :type rels: torch.FloatTensor
        :return: tuple of loss and predicted score
        :rtype: Tuple[torch.FloatTensor,torch.FloatTensor]
        """
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (ltensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        log_probs = torch.gather(outputs, 1, rels.unsqueeze(1))
        loss = self.nll(outputs, rels)
        preds = 0
        for j in range(0, self.num_relations):
            index = (ltensor([j])).cuda()
            """ j+1 because of zero index """
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return loss, preds


class NodeClassifier(nn.Module):
    """Multi-layer perceptron to predict node-level labels given embeddings

    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param embeddings: node-level embeddings
    :type embeddings: torch.FloatTensor
    """

    def __init__(self, embed_dim: int, embeddings: torch.FloatTensor):
        """Constructor method"""
        super(NodeClassifier, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        h = embed_dim

        self.age = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 7),
        )

        self.occupation = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 21),
        )

        self.gender = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 1),
        )

    def set_mode(self, mode: str) -> None:
        """Set mode of network

        :param mode: sensitive attribute we seek to defend/attack, one of (gender, age, occupation)
        :type mode: str
        """
        self.mode = mode

    def forward(
        self,
        features: Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
    ) -> Tuple[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
    ]:
        """Predict node-level label

        :param features: Encoded features
        :type features: Tuple[ torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor ]
        :return: predicted node-level label
        :rtype: Tuple[ torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], ]
        """
        (user, gender, occupation, age) = features

        embeddings = self.embeddings[user]

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if self.mode == "gender":
            loss = fn_gender(gender_pred, gender.float())
        elif self.mode == "age":
            loss = fn_age(age_pred, age)
        elif self.mode == "occupation":
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]


class SimpleGCMC(nn.Module):
    """Naive GCMC Encoder-Decoder architecture, as specified in https://arxiv.org/pdf/1706.02263.pdf

    :param decoder: decoder, usually SharedBilinearDecoder
    :type decoder: torch.nn.Module
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param num_ent: number of entities
    :type num_ent: int
    :param encoder: encoder neural network, if not using default embedding, defaults to None
    :type encoder: torch.nn.Module, optional
    """

    def __init__(
        self, decoder: torch.nn.Module, embed_dim: int, num_ent: int, encoder=None
    ):
        """Constructor method"""
        super(SimpleGCMC, self).__init__()
        self.decoder = decoder
        self.num_ent = num_ent
        self.embed_dim = embed_dim
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        if encoder is None:
            r = 6 / np.sqrt(self.embed_dim)
            self.encoder = nn.Embedding(
                self.num_ent, self.embed_dim, max_norm=1, norm_type=2
            )
            self.encoder.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)
        else:
            self.encoder = encoder
        self.all_nodes = to_device(ltensor(list(range(MAX_NUM_USERS))))

    def encode(self, nodes: Union[None, torch.LongTensor]) -> torch.FloatTensor:
        """Encode nodes using learned embeddings

        :param nodes: node identities
        :type nodes: Union[None, torch.LongTensor]
        :return: learnable embeddings of nodes
        :rtype: torch.FloatTensor
        """
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        return embs[nodes]

    def set_mode(self, mode: str) -> None:
        """Set mode of network

        :param mode: sensitive attribute we seek to defend/attack, one of (gender, age, occupation)
        :type mode: str
        """
        self.mode = mode

    def predict_rel(
        self, heads: Union[None, torch.LongTensor], tails_embed: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Predict scores based on head and tail embeddings of edges in a node/entries in a matrix

        :param heads: head embeddings
        :type heads: Union[None, torch.LongTensor]
        :param tails_embed: tail embeddings
        :type tails_embed: torch.FloatTensor
        :return: predicted scores
        :rtype: torch.FloatTensor
        """
        with torch.no_grad():
            head_embeds = self.encode(heads)
            preds = self.decoder.predict(head_embeds, tails_embed)
        return preds

    def forward(
        self, pos_edges: torch.FloatTensor, return_embeds=False
    ) -> Union[
        Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ]:
        """Predict scores of positive edges

        :param pos_edges: observed edges
        :type pos_edges: torch.FloatTensor
        :param return_embeds: whether to return head & tail embeddings, defaults to False
        :type return_embeds: bool, optional
        :return: Tuple of either loss and predictions, or loss, predictions and head & tail embeddings
        :rtype: Union[ Tuple[ torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor ], Tuple[torch.FloatTensor, torch.FloatTensor], ]
        """
        pos_head_embeds = self.encode(pos_edges[:, 0])
        pos_tail_embeds = self.encode(pos_edges[:, -1])
        rels = pos_edges[:, 1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds

    def save(self, fn: str) -> None:
        """Save model to a file

        :param fn: file location
        :type fn: str
        """
        torch.save(self.state_dict(), fn)

    def load(self, fn: str) -> None:
        """Load model from a file

        :param fn: file location
        :type fn: str
        """
        self.load_state_dict(torch.load(fn))


class GAL(SimpleGCMC):
    """Information Obfuscation, Node Level

    :param decoder: decoder neural network
    :type decoder: torch.nn.Module
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param num_ent: number of entities
    :type num_ent: int
    :param edges: edge tensor
    :type edges: torch.LongTensor
    :param args: namespace for input arguments
    :type args: argparse.Namespace
    :param encoder: encoder neural network, if not using default embedding, defaults to None
    :type encoder: torch.nn.Module, optional
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        embed_dim: int,
        num_ent: int,
        edges: torch.LongTensor,
        args: argparse.Namespace,
        encoder=None,
    ):
        """Constructor method"""
        super(GAL, self).__init__(decoder, embed_dim, num_ent, encoder=None)

        self.args = args
        self.gnn = GNN(self.embed_dim, args.gnn_layers, args.gnn_type, args.device)
        self.edges = edges.to(self.args.device)

        h = embed_dim

        self.age = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 7),
        )

        self.occupation = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 21),
        )

        self.gender = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 1),
        )

        self.reverse = GradientReversalLayer()

    def encode(self, nodes):
        """Encode nodes using graph neural networks

        :param nodes: node identities
        :type nodes: Union[None, torch.LongTensor]
        :return: learnable, graph-neural embeddings of nodes
        :rtype: torch.FloatTensor
        """
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        embs = self.gnn(embs, self.edges)
        return embs[nodes]

    def forward_attr(
        self,
        user_features: Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
    ) -> Tuple[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
    ]:
        """Compute adversarial loss with respect to user features

        :param user_features: user features
        :type user_features: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        :return: tuple of adversarial loss and predictions of three sensitive attributes (age, gender, occupation)
        :rtype: Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]
        """
        (users, gender, occupation, age) = user_features
        user_embeds = self.reverse(self.encode(users))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if self.mode == "gender":
            loss_adv = fn_gender(gender_pred, gender.float())
        elif self.mode == "age":
            loss_adv = fn_age(age_pred, age)
        elif self.mode == "occupation":
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)

    def set_mode(self, mode: str) -> None:
        """Set mode of network

        :param mode: sensitive attribute we seek to defend/attack, one of (gender, age, occupation)
        :type mode: str
        """
        self.mode = mode


class NeighborClassifier(nn.Module):
    """A special type of multi-layer perceptron that learns node-level labels from embeddings of its neighbors

    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param embeddings: node-level embeddings
    :type embeddings: torch.FloatTensor
    :param edges: edges
    :type edges: torch.LongTensor
    """

    def __init__(
        self, embed_dim: int, embeddings: torch.FloatTensor, edges: torch.LongTensor
    ):
        """Constructor method"""
        super(NeighborClassifier, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        h = embed_dim

        self.age = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 7),
        )

        self.occupation = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 21),
        )

        self.gender = Sequential(
            Linear(h, 32),
            LeakyReLU(),
            Linear(32, 32),
            LeakyReLU(),
            Linear(32, 1),
        )

        edges_np = edges.numpy()
        edge_list = []
        for i in range(edges_np.shape[1]):
            edge_list.append((edges_np[0, i], edges_np[1, i]))
        self.G = nx.Graph(edge_list)

    def set_mode(self, mode: str) -> None:
        """Set mode of network

        :param mode: sensitive attribute we seek to defend/attack, one of (gender, age, occupation)
        :type mode: str
        """
        self.mode = mode

    def forward(
        self,
        features: Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
    ):
        """Predict node-level label based on neighborhood-level information

        :param features: Encoded features
        :type features: Tuple[ torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor ]
        :return: predicted node-level label
        :rtype: Tuple[ torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], ]
        """
        (user, gender, occupation, age) = features

        neighbor = []

        user_np = list(user.cpu().numpy())

        for user in user_np:
            for i in self.G.neighbors(user):
                neighbor.append(i)
                break

        embeddings = self.embeddings[neighbor, :]

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if self.mode == "gender":
            loss = fn_gender(gender_pred, gender.float())
        elif self.mode == "age":
            loss = fn_age(age_pred, age)
        elif self.mode == "occupation":
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]


class GAL_Neighbor(GAL):
    """Information Obfuscation, Neighborhood Level

    :param decoder: decoder neural network
    :type decoder: torch.nn.Module
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param num_ent: number of entities
    :type num_ent: int
    :param edges: edge tensor
    :type edges: torch.LongTensor
    :param args: namespace for input arguments
    :type args: argparse.Namespace
    :param encoder: encoder neural network, if not using default embedding, defaults to None
    :type encoder: torch.nn.Module, optional
    """

    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None):
        """Constructor method"""
        super(GAL_Neighbor, self).__init__(
            decoder, embed_dim, num_ent, edges, args, encoder=None
        )

        edges_np = edges.numpy()
        edge_list = []
        for i in range(edges_np.shape[1]):
            edge_list.append((edges_np[0, i], edges_np[1, i]))
        self.G = nx.Graph(edge_list)
        self.args = args

    def forward_attr(
        self,
        user_features,
    ):
        (users, gender, occupation, age) = user_features

        neighbor = []

        user_np = list(users.cpu().numpy())

        for user in user_np:
            for i in self.G.neighbors(user):
                neighbor.append(i)
                break

        neighbor = torch.tensor(np.array(neighbor)).to(self.args.device)

        user_embeds = self.reverse(self.encode(neighbor))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if self.mode == "gender":
            loss_adv = fn_gender(gender_pred, gender.float())
        elif self.mode == "age":
            loss_adv = fn_age(age_pred, age)
        elif self.mode == "occupation":
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)


class GAL_Nhop(GAL):
    """Information Obfuscation, N-Hop Level

    :param decoder: decoder neural network
    :type decoder: torch.nn.Module
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param num_ent: number of entities
    :type num_ent: int
    :param edges: edge tensor
    :type edges: torch.LongTensor
    :param args: namespace for input arguments
    :type args: argparse.Namespace
    :param hop: number of hops to probabilistically sample
    :type hop: int, defaults to 2
    :param encoder: encoder neural network, if not using default embedding, defaults to None
    :type encoder: torch.nn.Module, optional
    """

    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None, hop=2):
        """Constructor method"""
        super(GAL_Nhop, self).__init__(
            decoder, embed_dim, num_ent, edges, args, encoder=None
        )

        edges_np = edges.numpy()
        edge_list = []
        for i in range(edges_np.shape[1]):
            edge_list.append((edges_np[0, i], edges_np[1, i]))
        self.G = nx.Graph(edge_list)
        self.hop = hop
        self.args = args

    def forward_attr(
        self,
        user_features,
    ):
        """Routine that implements probabilistic n-hop algorithm proposed in our paper.
        Intuitively, this algorithm greedily constructs a path of length n by uniformly picking a neighbor from the current end of the
        path and checking if the node has existed previously in the path, avoiding formation of cycles. Worst-case running time
        of this algorithm is O(n^2), because in each step of the main loop, the algorithm performs O(n) checks in the worst case scenario
        """
        (users, gender, occupation, age) = user_features

        neighbor_sub = []

        user_np = list(users.cpu().numpy())

        k = 0
        include_indices = []
        ign_indices = []
        ign_str = "failed vertices: "
        for user in user_np:
            final = user
            path = dict()
            for _ in range(self.hop):
                neighbor = list(self.G.neighbors(final))
                L = lambda a: len(a)
                orig_len = L(neighbor)

                cond = True
                cand = neighbor.pop(np.random.randint(0, L(neighbor)))
                if cand in path:

                    # This is bounded by O(self.hop)
                    while L(neighbor) > 0:
                        cand = neighbor.pop(np.random.randint(0, L(neighbor)))
                        if cand not in path:
                            break
                    if L(neighbor) == 0:
                        ign_str += "[{} <= {}]".format(orig_len, len(path))
                        cond = False
                        break

                if cond:
                    path[cand] = 0
                    final = cand
                else:
                    break
            if cond:
                neighbor_sub.append(final)
                include_indices.append(k)
            else:
                ign_indices.append(k)
            k += 1

        include_indices = np.array(include_indices)
        if len(ign_indices) > 0:
            warnings.warn(
                "ignoring {} from {}".format(ign_indices, ign_str),
                RuntimeWarning,
                stacklevel=2,
            )
        gender = gender[include_indices]
        age = age[include_indices]
        occupation = occupation[include_indices]

        neighbor = torch.tensor(np.array(neighbor_sub)).to(self.args.device)

        user_embeds = self.reverse(self.encode(neighbor))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if self.mode == "gender":
            loss_adv = fn_gender(gender_pred, gender.float())
        elif self.mode == "age":
            loss_adv = fn_age(age_pred, age)
        elif self.mode == "occupation":
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)
