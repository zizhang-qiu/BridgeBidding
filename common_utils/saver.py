"""
@file:saver
@author:qzz
@date:2023/3/21
@encoding:utf-8
"""
import os
from typing import List, Tuple

import torch


class TopKSaver:
    def __init__(self, k: int, save_dir: str, prefix: str):
        """
        A saver saves top k performance models
        Args:
            k: How many models to save
            save_dir: The directory to save files
            prefix: The file name's prefix
        """
        self.k = k
        self.topk: List[Tuple[float, object]] = []
        self.save_dir = save_dir
        self.prefix = prefix

    def _save_if_topk(self, obj: object, performance: float):
        # if full and the performance is smaller than the smallest in top k, return
        if len(self.topk) == self.k and performance < self.topk[-1][0]:
            return
        self.topk.append((performance, obj))
        self.topk.sort(key=lambda x: x[0], reverse=True)
        if len(self.topk) > self.k:
            del self.topk[-1]
            assert len(self.topk) == self.k

    def save(self, obj: object, performance: float):
        """
        Save the model with performance
        Args:
            obj: The object to save
            performance: The performance

        Returns:
            No returns
        """
        self._save_if_topk(obj, performance)
        perf_str = ""
        for i, (perf, net) in enumerate(self.topk):
            torch.save(net, os.path.join(self.save_dir, f"{self.prefix}_{i}.pth"))
            perf_str += f"{i}: {perf}\n"
        with open(os.path.join(self.save_dir, "performances.txt"), "w") as f:
            f.write(perf_str)
