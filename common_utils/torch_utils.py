#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:torch_utils.py
@time:2023/02/17
"""
from typing import List

import torch
import torch.multiprocessing as mp
from torch import nn


def print_parameters(model: nn.Module):
    """
    Print a network's parameter

    Args:
        model (nn.Module): the model to print

    Returns:
        No returns
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_device(model: nn.Module):
    """
    Returns the device on which a PyTorch model is located.

    Args:
        model: A PyTorch model.

    Returns:
        A string representing the device on which the model is located,
        e.g. 'cpu' or 'cuda:0'.
    """
    return str(next(model.parameters()).device)


def reset_network_params(model):
    for param in model.parameters():
        param.data.fill_(0)


def create_shared_dict(net: nn.Module):
    net_state_dict = net.state_dict()
    for k, v in net_state_dict.items():
        net_state_dict[k] = v.cpu()

    shared_dict = mp.Manager().dict()
    shared_dict.update(net_state_dict)

    return shared_dict


def clone_parameters(net: nn.Module) -> List[torch.Tensor]:
    cloned_parameters = [p.clone() for p in net.parameters()]
    return cloned_parameters


def check_updated(parameters_before_update: List[torch.Tensor], parameters_after_update: List[torch.Tensor]):
    assert len(parameters_before_update) == len(parameters_after_update)
    for p_b, p_a in zip(parameters_before_update, parameters_after_update):
        if not torch.equal(p_b, p_a):
            return True
    return False


def initialize_fc(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.01)


def to_device(*args, device="cuda"):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return tuple(ret)
