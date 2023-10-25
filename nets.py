#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:nets.py
@time:2023/03/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge_state import AUCTION_TENSOR_SIZE
from bridge_vars import NUM_CALLS


class MLP(nn.Module):
    def __init__(self):
        """The network used for the paper 'Human-Agent Cooperation in Bridge Bidding'."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUCTION_TENSOR_SIZE, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, NUM_CALLS)
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.log_softmax(out, -1)
        return policy
