import json
from itertools import product

modes = ["Fixed_Embedding", "Naive_GNN", "GAL"]
seeds = [0, 42]
lambdas = [0.5, 0.9]
hops = [3, 4, 8, 16]
tasks = ["Neighbor_Attack", "Node_Attack"] + list(product(["NHop_Attack"], hops))

d = {
    "task": "Neighbor_Attack",
    "model": "GAL",
    "experiment": "ablation",
    "num_epochs": 2,
    "batch_size": 8192,
    "node_cls_batch_size": 256,
    "valid_freq": 1,
    "embed_dim": 20,
    "lr": 0.01,
    "prefetch_to_gpu": 0,
    "seed": 11,
    "debug": 0,
    "gnn_layers": 3,
    "gnn_type": "ChebConv",
    "finetune_epochs": 3,
    "data_path": "/",
    "cutoff": 0.8,
    "lambda_reg": 0.5,
    "device_num": 0,
    "hop": 3,
}


def generate_training_traces():
    for (mode, seed, task, lam) in product(modes, seeds, tasks, lambdas):
        if len(task) == 2:
            task, hop = task
            with open(f"{mode}_{task}_{hop}hop_{lam}_{seed}.json", "w") as outfile:
                d["model"] = mode
                d["seed"] = seed
                d["task"] = task
                d["lambda_reg"] = lam
                d["hop"] = hop
                json.dump(d, outfile)
        else:
            with open(f"{mode}_{task}_{lam}_{seed}.json", "w") as outfile:
                d["model"] = mode
                d["seed"] = seed
                d["task"] = task
                d["lambda_reg"] = lam
                json.dump(d, outfile)


if __name__ == "__main__":
    generate_training_traces()
