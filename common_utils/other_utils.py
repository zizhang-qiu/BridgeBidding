#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:other_utils.py
@time:2023/02/19
"""
import datetime
import os
import pickle
import platform
from typing import Any, List

import numpy as np
import random
import torch


def set_random_seeds(seed: int = 42):
    """Set random seeds for PyTorch and NumPy."""
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python built-in
    random.seed(seed)


def check_picklable(obj: Any):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def gcd(a: int, b: int) -> int:
    """Returns the greatest common divisor of two integers a and b."""
    while b:
        a, b = b, a % b
    return a


def lcm(numbers: List[int]) -> int:
    """Returns the least common multiple of a list of integers."""
    if len(numbers) == 0:
        return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = (result * numbers[i]) // gcd(result, numbers[i])
    return result


def set_omp_threads(num_threads: int = 1, verbose: bool = False) -> None:
    """
    set openmp thread to num_thread

    Args:
        num_threads: (int) the number of threads, default 1
        verbose: (bool) whether to print message

    Returns:None

    """
    system_platform = platform.system().lower()
    if system_platform.startswith("w"):
        os.system(f"set OMP_NUM_THREADS={num_threads}")
    elif system_platform.startswith("l"):
        os.system(f"export OMP_NUM_THREADS={num_threads}")
    if verbose:
        print(f"set omp_num_thread={num_threads}")


def mkdir_with_time(_dir: str, _format: str = '%Y%m%d%H%M%S') -> str:
    """
    Make directory using time strf, return the path to the directory.
    Args:
        _dir(str):the directory to make a new directory
        _format(str): the strf format of time

    Returns:
        str: the path to the made directory
    """
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    time_str = datetime.datetime.now().strftime(_format)
    path = os.path.join(_dir, time_str)
    os.mkdir(path)
    return path
