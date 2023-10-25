#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:make_dataset.py
@time:2023/10/25
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from bridge_state import BridgeBiddingState, encode_observation_tensor
from bridge_vars import NUM_CARDS, NUM_PLAYERS
from global_vars import Action


def parse_args():
    parser = argparse.ArgumentParser(description="args for supervised learning.")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--save_dir", type=str, default="supervised_learning_dataset")

    return parser.parse_args()


def make_dataset_from_file(f: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(f, "r") as f:
        content = f.readlines()
    obs = []
    labels = []
    for line in tqdm(content):
        trajectory = _no_play_trajectory(line)
        state = BridgeBiddingState(False, False, np.zeros(20))
        # Deal.
        for i in range(NUM_CARDS):
            state.apply_action(trajectory[i])

        for i in range(NUM_CARDS, len(trajectory)):
            s = encode_observation_tensor(state)
            label = trajectory[i] - NUM_CARDS
            obs.append(s)
            labels.append(label)
            state.apply_action(trajectory[i])

    obs = torch.stack(obs, 0)
    labels = torch.tensor(labels)
    return obs, labels


def _no_play_trajectory(line: str) -> List[Action]:
    """Get trajectory without playing phase"""
    actions = [int(action) for action in line.split(" ")]
    # print(actions)
    if len(actions) == NUM_CARDS + NUM_PLAYERS:
        # All player passes
        return actions
    else:
        # Otherwise delete the playing phase, which has NUM_CARDS actions
        return actions[:-NUM_CARDS]


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    sources = []
    for root, folder, files in os.walk(args.dataset_dir):
        for file in files:
            if file.endswith(".txt"):
                sources.append(os.path.join(root, file))

    for s in sources:
        print(f"Creating dataset for {os.path.split(s)[-1].split('.')[0]}.")
        obs, labels = make_dataset_from_file(s)
        save_dict = {
            "obs": obs,
            "labels": labels
        }
        torch.save(save_dict, os.path.join(args.save_dir, os.path.split(s)[-1].split(".")[0]+".pth"))
