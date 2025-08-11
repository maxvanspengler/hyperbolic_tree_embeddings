import torch.nn as nn
from torch.optim import Optimizer


def clone_one_group_optimizer(
    optimizer: Optimizer, new_params: nn.Parameter, **kwargs
) -> Optimizer:
    opt_class = optimizer.__class__
    opt_dict = optimizer.param_groups[0].copy()
    opt_dict.pop("params")

    for k, v in kwargs.items():
        assert k in opt_dict, (
            f"During cloning, key {k} was supplied, but not previously present in {optimizer}"
        )
        opt_dict[k] = v

    return opt_class(
        params=new_params,
        **opt_dict
    )
