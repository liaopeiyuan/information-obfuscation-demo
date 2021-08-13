# type: ignore
# -*- coding: utf-8 -*-

import adabound
import torch.optim as optim


def create_optimizer(params, mode: str, *args, **kwargs) -> optim.Optimizer:
    """Create an optimizer corresponding to the params

    :param params: parameters
    :type params: iterable
    :param mode: type of optimizers
    :type mode: str
    :raises NotImplementedError: type of optimizer is not recognized
    :return: optimizer corresponding to the params
    :rtype: torch.optim.Optimizer
    """
    if mode == "SGD":
        opt = optim.SGD(params, *args, momentum=0.0, **kwargs)
    elif mode.startswith("nesterov"):
        momentum = float(mode[len("nesterov") :])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == "adam":
        betas = kwargs.pop("betas", (0.9, 0.999))
        opt = optim.Adam(
            params, *args, betas=betas, amsgrad=True, weight_decay=1e-4, **kwargs
        )
    elif mode.lower() == "adam_hyp2":
        betas = kwargs.pop("betas", (0.5, 0.99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == "adam_hyp3":
        betas = kwargs.pop("betas", (0.0, 0.99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == "adam_sparse":
        betas = kwargs.pop("betas", (0.9, 0.999))
        opt = optim.SparseAdam(params, *args, weight_decay=1e-4, betas=betas)
    elif mode.lower() == "adam_sparse_hyp2":
        betas = kwargs.pop("betas", (0.5, 0.99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == "adam_sparse_hyp3":
        betas = kwargs.pop("betas", (0.0, 0.99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == "adabound":
        opt = adabound.AdaBound(params, *args, final_lr=0.1)
    else:
        raise NotImplementedError()
    return opt
