#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:train.py
@time:2023/10/25
"""
import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader

from bridge_vars import NUM_CALLS
from nets import MLP


class BridgeBiddingDataset(Dataset):
    def __init__(self, file: str):
        d = torch.load(file)
        self.obs: torch.Tensor = d["obs"]
        self.labels: torch.Tensor = d["labels"]

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, item):
        return self.obs[item], self.labels[item]


def parse_args():
    parser = argparse.ArgumentParser(description="args for supervised learning.")
    parser.add_argument("--dataset_dir", type=str, default="supervised_learning_dataset")
    parser.add_argument("--save_dir", type=str, default="supervised_learning_mlp")

    # train parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--valid_batch_size", type=int, default=10000)
    parser.add_argument("--num_episodes", type=int, default=500000, help="How many iterations to train.")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation.")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def cross_entropy(log_probs: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute cross entropy loss of given log probs and label.
    Args:
        log_probs: The log probs.
        label: The label, should be 1 dimensional.
        num_classes: The number of classes for one-hot.

    Returns:
        The cross entropy loss.
    """
    assert label.ndimension() == 1
    return -torch.mean(torch.nn.functional.one_hot(label.long(), num_classes) * log_probs)


def compute_accuracy(probs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy of given probs and label. Which is the number of highest value action equals with label
    divides number of all actions.
    Args:
        probs: The probs.
        label: The labels.

    Returns:
        The accuracy of prediction.
    """
    greedy_actions = torch.argmax(probs, 1)
    return (greedy_actions == label).int().sum() / greedy_actions.shape[0]


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    train_dataset = BridgeBiddingDataset(os.path.join(args.dataset_dir, "valid.pth"))
    valid_dataset = BridgeBiddingDataset(os.path.join(args.dataset_dir, "valid.pth"))
    train_dataloader = DataLoader(train_dataset, args.train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, args.valid_batch_size)

    net = MLP()
    net.to(args.device)
    opt = torch.optim.Adam(lr=args.learning_rate, params=net.parameters())

    num_mini_batches = 0

    while num_mini_batches < args.num_episodes:
        for s, labels in train_dataloader:
            num_mini_batches += 1
            opt.zero_grad()
            log_probs = net(s.to(args.device))
            loss = cross_entropy(log_probs, labels.to(args.device), NUM_CALLS)
            loss.backward()
            opt.step()

            if num_mini_batches % args.eval_freq == 0:
                loss_list = []
                acc_list = []
                with torch.no_grad():
                    for s, labels in valid_dataloader:
                        log_probs = net(s.to(args.device))
                        labels = labels.to(args.device)
                        loss = torch.nn.functional.one_hot(labels.long(), NUM_CALLS) * log_probs
                        loss_list.append(loss)
                        greedy_actions = torch.argmax(log_probs, 1)
                        acc = (greedy_actions == labels).float()
                        acc_list.append(acc)
                loss = torch.cat(loss_list, 0)
                acc = torch.cat(acc_list)
                print(
                    f"Epoch {num_mini_batches // args.eval_freq}, loss={-loss.mean().item()}, acc={acc.mean().item()}")
                torch.save(net.state_dict(),
                           os.path.join(args.save_dir, f"net_{num_mini_batches // args.eval_freq}.pth"))
