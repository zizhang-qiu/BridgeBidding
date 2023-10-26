"""Utils for assertions"""
from typing import NoReturn, Iterable

import numpy as np
import torch
from torch import nn


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_neq(real, expected):
    assert real != expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_lt(real, expected):
    assert real < expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_lteq(real, expected):
    assert real <= expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_tensor_eq(t1, t2, eps=1e-6):
    if t1.size() != t2.size():
        print("Warning: size mismatch", t1.size(), "vs", t2.size())
        return False

    t1 = t1.cpu().numpy()
    t2 = t2.cpu().numpy()
    diff = abs(t1 - t2)
    eq = (diff < eps).all()
    if not eq:
        import pdb

        pdb.set_trace()
    assert eq


def assert_zero_grad(params):
    for p in params:
        if p.grad is not None:
            assert p.grad.sum().item() == 0


def assert_in(item, obj: Iterable):
    assert item in obj, f"item {item} not in iterable {obj}."


def assert_in_range(real, range_left, range_right) -> NoReturn:
    """
    assert a num in a left closed right open range interval
    Args:
        real: the real number
        range_left: the left range, it is closed
        range_right: the right range, it is open

    Returns:
        No return
    """
    assert range_left <= real < range_right, f"expected range is [{range_left}, {range_right}), the number is {real}"


def assert_not_inf_nan(real):
    """assert the number is not inf or nan, real should be one number or ndarray(tensor) contains one element"""
    if isinstance(real, torch.Tensor):
        real_ = real.detach().cpu().numpy()
        shape = real_.shape
        assert len(shape) == 0 or len(shape) == 1, f"the input should be one number, but got {real}."
        # ndarray(0) like
        if len(shape) == 1:
            real_ = real_.item()
            if np.isnan(real_):
                assert False, "the number is nan"
            if np.isinf(real_):
                assert False, "the number is inf"
    elif isinstance(real, np.ndarray):
        shape = real.shape
        assert len(shape) == 0 or len(shape) == 1, f"the input should be one number, but got {real}."
        # ndarray(0) like
        if len(shape) == 1:
            real_ = real.item()
            if np.isnan(real_):
                assert False, "the number is nan"
            if np.isinf(real_):
                assert False, "the number is inf"
    elif isinstance(real, int) or isinstance(real, float):
        if np.isnan(real):
            assert False, "the number is nan"
        if np.isinf(real):
            assert False, "the number is inf"
    else:
        raise TypeError(f"the available type is torch.Tensor, np.ndarray, int and float, but got {type(real)}")


def assert_network_normal(network: nn.Module):
    for param in network.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            assert False, "the network contains nan or inf!"
