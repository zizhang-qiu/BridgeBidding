#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:bridge_scoring.py
@time:2023/03/22
"""
from bisect import bisect

from bridge_vars import Denomination, DoubleStatus, DENOMINATION_STR, PLAYER_STR, NUM_DENOMINATIONS, \
    NUM_PLAYERS, NUM_DOUBLE_STATES


class Contract:
    """The contract class

    This class can track a contract's level, trump, double status and declarer.

    Attributes:
        level: A integer represents the level of the contract
        trumps: An attribute of Denomination
        double status: An attribute of DoubleStatus, undoubled, doubled or redoubled
        declarer: A integer represent declarer
    """

    def __init__(self, level: int = 0, trump: Denomination = Denomination.NO_TRUMP,
                 double_status: DoubleStatus = DoubleStatus.UNDOUBLED, declarer: int = -1):
        self.level: int = level
        self.trumps: Denomination = trump
        self.double_status: DoubleStatus = double_status
        self.declarer: int = declarer

    def __repr__(self):
        if self.level == 0:
            return "Passed Out"
        ret = str(self.level) + DENOMINATION_STR[self.trumps.value]
        if self.double_status == DoubleStatus.DOUBLED:
            ret += "X"
        if self.double_status == DoubleStatus.REDOUBLED:
            ret += "XX"
        ret += f" {PLAYER_STR[self.declarer]}"
        return ret

    def index(self) -> int:
        """
        convert a contract to index, contracts is ordered like

        Passed Out, 1C N, 1CX N, 1CXX N, 1C E ,... ,1D N, ... 7NXX W

        In a word, level and trump goes first, then double status, finally the player

        level and trump goes as 1C, 1D, 1H, 1S, 1N,

        double status goes like ""(undoubled), X, XX

        and player goes like NESW, clockwise start from N

        there are 1 + (35 * 3 * 4) = 421 contracts and "Passed Out" is indexed as 0

        Returns:
            int: the index

        """
        if self.level == 0:
            return 0
        index = self.level - 1
        index *= NUM_DENOMINATIONS
        index += self.trumps.value
        index *= NUM_PLAYERS
        index += self.declarer
        index *= NUM_DOUBLE_STATES
        if self.double_status == DoubleStatus.REDOUBLED:
            index += 2
        if self.double_status == DoubleStatus.DOUBLED:
            index += 1
        return index + 1


def make_all_contracts():
    ret = [Contract()]
    for level in [1, 2, 3, 4, 5, 6, 7]:
        for trump in Denomination:
            for declarer in range(NUM_PLAYERS):
                for double_status in DoubleStatus:
                    ret.append(Contract(level, trump, double_status, declarer))
    return ret


all_contracts = make_all_contracts()
# print(all_contracts)

base_trick_scores = [20, 20, 30, 30, 30]


def score_contract(contract: Contract, double_status: DoubleStatus) -> int:
    score = contract.level * base_trick_scores[contract.trumps]
    if contract.trumps == Denomination.NO_TRUMP:
        score += 10
    return score * double_status


def score_undertricks(undertricks: int, is_vulnerable: bool, double_status: DoubleStatus) -> int:
    if double_status == DoubleStatus.UNDOUBLED:
        return (-100 if is_vulnerable else -50) * undertricks
    if is_vulnerable:
        score = -200 - 300 * (undertricks - 1)
    else:
        if undertricks == 1:
            score = -100
        elif undertricks == 2:
            score = -300
        else:
            score = -500 - 300 * (undertricks - 3)

    return score * (double_status // 2)


def score_overtricks(trump_suit: Denomination, overtricks: int, is_vulnerable: bool,
                     double_status: DoubleStatus) -> int:
    if double_status == DoubleStatus.UNDOUBLED:
        return overtricks * base_trick_scores[trump_suit]
    else:
        return (100 if is_vulnerable else 50) * overtricks * double_status


def score_doubled_bonus(double_status: DoubleStatus) -> int:
    return 50 * (double_status // 2)


def score_bonuses(level: int, contract_score: int, is_vulnerable: bool) -> int:
    if level == 7:  # grand slam
        return 200 if is_vulnerable else 1300
    if level == 6:  # slam
        return 1250 if is_vulnerable else 800
    if contract_score >= 100:
        return 500 if is_vulnerable else 300
    return 50


def compute_score(contract: Contract, declarer_tricks: int, is_vulnerable: bool):
    """
    Get duplicate score with contract, declarer tricks and vulnerability
    Args:
        contract (Contract):the contract
        declarer_tricks (int): how many tricks declarer made, it should be real tricks not bid level
        is_vulnerable (bool): whether the declarer is vulnerable

    Returns:
        int: the score for declarer

    """
    if contract.level == 0:
        return 0
    contracted_tricks = 6 + contract.level  # how many tricks should be made
    contract_result = declarer_tricks - contracted_tricks  # difference between real tricks and needed tricks
    if contract_result < 0:
        return score_undertricks(-contract_result, is_vulnerable, contract.double_status)
    else:
        contract_score = score_contract(contract, contract.double_status)
        bonuses = (score_bonuses(contract.level, contract_score, is_vulnerable) +
                   score_doubled_bonus(contract.double_status) +
                   score_overtricks(contract.trumps, contract_result, is_vulnerable, contract.double_status))
        return contract_score + bonuses


def get_imp(my: int, other: int) -> int:
    """Function to get imp

    Args:
        my (int): your duplicate score at ns
        other (int): other team's duplicate score at ns

    Returns:
        int: imp this board

    """
    imp_table = [
        15, 45, 85, 125, 165, 215, 265, 315, 365, 425, 495, 595, 745, 895,
        1095, 1295, 1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995]
    return bisect(imp_table, abs(my - other)) * (1 if my > other else -1)
