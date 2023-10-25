#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:bridge_state.py
@time:2023/03/22
"""

from collections import Counter
from enum import IntEnum
from typing import Optional, List

import numpy as np
import torch

from bridge_scoring import Contract, compute_score
from bridge_vars import NUM_CARDS, NUM_SUITS, Suit, SUIT_STR, RANK_STR, NUM_CARDS_PER_SUIT, \
    NUM_PLAYERS, NUM_OTHER_CALLS, NUM_DENOMINATIONS, Denomination, LEVEL_STR, DENOMINATION_STR, DoubleStatus, \
    NUM_CONTRACTS, NUM_PARTNERSHIPS, NUM_BID_LEVELS, NUM_CALLS, PBN_PREFIX, PBN_TEMPLATE, NUM_VULNERABILITIES
from common_utils.assert_utils import assert_eq, assert_lteq, assert_neq, assert_lt, assert_in
from global_vars import PlayerAction, Action, PlayerId, Vector


class Phase(IntEnum):
    DEAL = 0
    AUCTION = 1
    GAME_OVER = 2


class BridgePlayer(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Calls(IntEnum):
    PASS = 0
    DOUBLE = 1
    REDOUBLE = 2


FIRST_BID = 3
BIDDING_ACTION_BASE = 52
balanced_hands = [[4, 3, 3, 3], [5, 3, 3, 2], [4, 4, 3, 2]]
balanced_hands_counters = [Counter({4: 1, 3: 3}), Counter({5: 1, 3: 2, 2: 1}), Counter({4: 2, 3: 1, 2: 1})]


# There are two partnerships: players 0 and 2 versus players 1 and 3.
# We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
def get_partnership(player: int):
    return player & 1


def get_partner(player: int):
    return player ^ 2


def get_bid_level(bid: int) -> int:
    return 1 + (bid - NUM_OTHER_CALLS) // NUM_DENOMINATIONS


def get_bid_suit(bid: int) -> Denomination:
    return Denomination((bid - NUM_OTHER_CALLS) % NUM_DENOMINATIONS)


def get_bid(level: int, denomination: Denomination) -> int:
    return (level - 1) * NUM_DENOMINATIONS + denomination + FIRST_BID


def get_bid_string(bid: int) -> str:
    if bid == Calls.PASS:
        return "Pass"
    if bid == Calls.DOUBLE:
        return "Dbl"
    if bid == Calls.REDOUBLE:
        return "RDbl"
    return LEVEL_STR[get_bid_level(bid)] + DENOMINATION_STR[get_bid_suit(bid)]


def get_card_suit(card: int) -> Suit:
    return Suit(card % NUM_SUITS)


def get_card_rank(card: int) -> int:
    """
    get the rank of the card, 0 represents "2"

    Args:
        card: the card

    Returns: the rank

    """
    return card // NUM_SUITS


def get_card(suit: Suit, rank: int):
    return rank * NUM_SUITS + suit


def get_card_string(card: int):
    return SUIT_STR[get_card_suit(card)] + RANK_STR[get_card_rank(card)]


class BridgeBiddingState:
    """A bidding state, assume start from North"""

    def __init__(self, is_dealer_vulnerable=False, is_non_dealer_vulnerable=False, ddt: Optional[Vector] = None):

        # vulnerability[0] represents dealer's vulnerability, [1] represents non-dealer's
        self._vulnerability = [is_dealer_vulnerable, is_non_dealer_vulnerable]
        self._ddt: Optional[np.ndarray] = None
        if ddt is not None:
            assert_eq(len(ddt), 5 * 4)
            self._ddt = np.array(ddt)
        self._current_player = PlayerId.CHANCE_PLAYER
        self._full_history: List[PlayerAction] = []
        self._phase = Phase.DEAL
        self._contract = Contract()
        self._returns = np.zeros(NUM_PLAYERS)
        self._possible_contracts = np.ones(NUM_CONTRACTS, dtype=bool)

        # track each card's holder
        self._holder = np.full(NUM_CARDS, fill_value=-1)

        # track consecutive passes
        self._num_passes = 0
        self._num_declarer_tricks = 0

        self._first_bidder = np.full([NUM_PARTNERSHIPS, NUM_DENOMINATIONS], fill_value=-1)

    def _compute_double_dummy_table(self):
        from dds import calc_dd_table
        self._ddt = calc_dd_table(self._holder).flatten()

    def apply_action(self, action: Action):
        self._do_apply_action(action)
        self._full_history.append(PlayerAction(self._current_player, action))

    def apply_action_with_legality_check(self, action: Action):
        legal_actions = self.legal_actions()
        assert_in(action, legal_actions)
        self.apply_action(action)

    def _do_apply_action(self, action):
        if self._phase == Phase.DEAL:
            self._apply_deal_action(action)
            return
        elif self._phase == Phase.AUCTION:
            action = action - BIDDING_ACTION_BASE if action >= BIDDING_ACTION_BASE else action
            self._apply_bidding_action(action)
            return
        else:
            print("Cannot act in terminal states")

    def compute_hand_evaluation(self) -> np.ndarray:
        """
        the hand evaluation for each player consist of length of each suit(4),
        the HCP(1), the suit length points(1), sui shortness points(1), support points(1),
        controls(1), is balanced(1),
        """
        hand_evaluation = np.zeros([NUM_PLAYERS, 10])
        cards_per_player: List[List[int]] = [[], [], [], []]
        for i, player in enumerate(self._holder):
            cards_per_player[player].append(i)

        for player in range(NUM_PLAYERS):
            player_cards = cards_per_player[player]
            cards_ranks = [get_card_rank(card) for card in player_cards]
            cards_suits = [get_card_suit(card) for card in player_cards]

            controls = 0

            # length of suits
            for suit in cards_suits:
                hand_evaluation[player][suit] += 1

            # HCP
            hcp = 0
            for rank in cards_ranks:
                hcp += max(0, rank - 8)
                controls += max(0, rank - 10)
            hand_evaluation[player][4] = hcp

            # suit length points
            suit_length_points = np.clip(hand_evaluation[player][:4] - 4, a_min=0,
                                         a_max=NUM_CARDS_PER_SUIT - 4).sum()
            hand_evaluation[player][5] = suit_length_points + hcp

            # suit shortness points
            suit_shortness_points = np.clip(-hand_evaluation[player][:4] + 3, a_min=0, a_max=3).sum()
            hand_evaluation[player][6] = suit_shortness_points + hcp

            # support points
            support_points = np.clip(-2 * hand_evaluation[player][:4] + 5, a_min=0, a_max=5).sum()
            hand_evaluation[player][7] = support_points + hcp

            # controls
            hand_evaluation[player][8] = controls

            # is balanced
            lengths_of_suits = hand_evaluation[player][:4].astype(int)
            # print(lengths_of_suits)
            lengths_counter = Counter(lengths_of_suits)
            # print(lengths_counter)
            for balanced_hand in balanced_hands_counters:
                if balanced_hand == lengths_counter:
                    hand_evaluation[player][9] = 1
                    break
        return hand_evaluation

    def _apply_deal_action(self, card: Action):
        self._holder[card] = len(self._full_history) % NUM_PLAYERS
        if len(self._full_history) == (NUM_CARDS - 1):
            self._phase = Phase.AUCTION
            self._current_player = BridgePlayer.NORTH
            if not isinstance(self._ddt, np.ndarray):
                self._compute_double_dummy_table()

    def _apply_bidding_action(self, call: Action):
        if call == Calls.PASS:
            self._num_passes += 1
        else:
            self._num_passes = 0

        partnership = get_partnership(self._current_player)

        # pass, double, redouble
        if call == Calls.DOUBLE:
            # can't double partner's bid
            assert_neq(get_partnership(self._contract.declarer), partnership)
            # can't double if the contract is already doubled
            assert_eq(self._contract.double_status, DoubleStatus.UNDOUBLED)
            # can't double if there is no contract except pass
            assert_lt(0, self._contract.level)
            self._possible_contracts[self._contract.index()] = False
            self._contract.double_status = DoubleStatus.DOUBLED

        elif call == Calls.REDOUBLE:
            # can only redouble partner's contract
            assert_eq(get_partnership(self._contract.declarer), partnership)
            # can only redouble if opponent doubled
            assert_eq(self._contract.double_status, DoubleStatus.DOUBLED)
            self._possible_contracts[self._contract.index()] = False
            self._contract.double_status = DoubleStatus.REDOUBLED

        elif call == Calls.PASS:
            # 4 consecutive passes makes the game passes out
            if self._num_passes == 4:
                self._phase = Phase.GAME_OVER
                self._possible_contracts.fill(False)
                self._possible_contracts[0] = True
            # 3 passes makes contract and end the game
            elif self._num_passes == 3 and self._contract.level > 0:
                self._possible_contracts.fill(False)
                self._possible_contracts[self._contract.index()] = True
                self._phase = Phase.GAME_OVER
                self._current_player = PlayerId.TERMINAL_PLAYER
                self._num_declarer_tricks = self._ddt[self._contract.trumps * NUM_PLAYERS
                                                      + self._contract.declarer]
                self.score_up()

        # a bid called
        else:
            # a bid should have higher suit ot higher level with same suit with contract bid before
            level = get_bid_level(call)
            suit = get_bid_suit(call)
            assert level > self._contract.level or \
                   (level == self._contract.level and suit > self._contract.trumps)

            self._contract.level = level
            self._contract.trumps = suit
            # a bid called, so reset to undoubled
            self._contract.double_status = DoubleStatus.UNDOUBLED

            if self._first_bidder[partnership][self._contract.trumps] == -1:
                # track who bid the trump first, the first bidder will be the declarer
                self._first_bidder[partnership][self._contract.trumps] = self._current_player
                partner = get_partner(self._current_player)
                # partner will not be the declarer of the trump
                for level in range(self._contract.level + 1, NUM_BID_LEVELS):
                    for double_status in DoubleStatus:
                        self._possible_contracts[
                            Contract(level, self._contract.trumps, double_status, partner).index()] = False

            self._contract.declarer = self._first_bidder[partnership][self._contract.trumps]

            # No lower contract is possible.
            self._possible_contracts[:Contract(self._contract.level, self._contract.trumps,
                                               DoubleStatus.UNDOUBLED, 0).index()] = False

            # No-one else can declare this precise contract.
            for player in range(NUM_PLAYERS):
                if player != self._current_player:
                    for double_status in DoubleStatus:
                        self._possible_contracts[Contract(self._contract.level, self._contract.trumps,
                                                          double_status, player).index()] = False

        self._current_player = (self._current_player + 1) % NUM_PLAYERS

    def score_up(self):
        """Method to compute scores and set to returns"""
        declarer_score = compute_score(self._contract,
                                       self._num_declarer_tricks,
                                       self._vulnerability[get_partnership(self._contract.declarer)])
        for pl in range(NUM_PLAYERS):
            self._returns[pl] = declarer_score if get_partnership(pl) == get_partnership(self._contract.declarer) \
                else -declarer_score

    def legal_actions(self) -> List[Action]:
        """
        Get legal actions for current state

        Returns: A list of legal actions

        """
        if self._phase == Phase.DEAL:
            return self._deal_legal_actions()
        if self._phase == Phase.AUCTION:
            return self._bidding_legal_actions()
        else:
            return []

    def _deal_legal_actions(self) -> List[Action]:
        legal_actions = [i for i in range(NUM_CARDS) if self._holder[i] == -1]
        return legal_actions

    def _bidding_legal_actions(self) -> List[Action]:
        # pass is always legal
        legal_actions = [BIDDING_ACTION_BASE + Calls.PASS]

        declarer_partnership = get_partnership(self._contract.declarer)
        current_player_partnership = get_partnership(self._current_player)
        contract_double_status = self._contract.double_status
        contract_level = self._contract.level

        # double is legal if the contract isn't bid by current player's partnership and not doubled
        if contract_level > 0 and declarer_partnership != current_player_partnership \
                and contract_double_status == DoubleStatus.UNDOUBLED:
            legal_actions.append(BIDDING_ACTION_BASE + Calls.DOUBLE)

        # redouble is legal if the contract is bid by current player's partnership and double by opponent partnership
        if contract_level > 0 and declarer_partnership == current_player_partnership \
                and contract_double_status == DoubleStatus.DOUBLED:
            legal_actions.append(BIDDING_ACTION_BASE + Calls.REDOUBLE)

        # any higher bid is legal
        for bid in range(get_bid(contract_level, self._contract.trumps) + 1, NUM_CALLS):
            legal_actions.append(BIDDING_ACTION_BASE + bid)

        return legal_actions

    def _original_deal(self) -> np.ndarray:
        assert_lteq(NUM_CARDS, len(self._full_history))
        deal = np.full(NUM_CARDS, -1)
        for i in range(NUM_CARDS):
            deal[self._full_history[i].action] = i % NUM_PLAYERS
        return deal

    def terminated(self) -> bool:
        return self._phase == Phase.GAME_OVER

    def _get_pbn_deal(self) -> str:
        deal_str = "N:"
        for player in range(NUM_PLAYERS):

            cards = ["", "", "", ""]
            for suit in range(NUM_SUITS):
                for rank in range(NUM_CARDS_PER_SUIT - 1, -1, -1):
                    if player == self._holder[get_card(Suit(suit), rank)]:
                        cards[suit] += RANK_STR[rank]
            cards.reverse()
            deal_str += ".".join(cards)
            deal_str += " "

        return deal_str[:-1]

    # todo: add bidding in pbn
    def to_pbn_form(self, with_bidding: bool = False):
        """We don't convert ddt to pbn form"""
        # converting a state when deal not end makes no sense
        assert self._phase > Phase.DEAL
        pbn_deal = self._get_pbn_deal()
        pbn_str = PBN_TEMPLATE.format(dealer="N", deal=pbn_deal)
        return pbn_str

    def write_into_pbn_file(self, file_path):
        pbn_str = self.to_pbn_form()
        with open(file_path, mode="r") as f:
            content = f.read()
        f.close()
        # print(content)

        with open(file_path, mode="a+") as f:
            if content:
                f.write("\n\n" + pbn_str)
            else:
                f.write(PBN_PREFIX + pbn_str)
        f.close()

    def _format_hand(self, player: int, mark_voids: bool, deal: np.ndarray) -> List[str]:
        """format a player's hand"""
        cards = ["", "", "", ""]
        for suit in range(NUM_SUITS):

            cards[suit] += SUIT_STR[suit]
            cards[suit] += ' '
            is_void = True
            for rank in range(NUM_CARDS_PER_SUIT - 1, -1, -1):
                if player == deal[get_card(Suit(suit), rank)]:
                    cards[suit] += RANK_STR[rank]
                    is_void = False
            if is_void and mark_voids:
                cards[suit] = cards[suit] + "none"
        return cards

    def _format_deal(self) -> str:
        cards = [[], [], [], []]

        for player in BridgePlayer:
            cards[player] = self._format_hand(player, False, self._holder)

        column_width = 8
        padding = " " * column_width
        rv = ""
        for suit in range(NUM_SUITS - 1, -1, -1):
            rv += padding + cards[BridgePlayer.NORTH][suit] + "\n"
        for suit in range(NUM_SUITS - 1, -1, -1):
            rv += f"{cards[BridgePlayer.WEST][suit]:8}" + padding + cards[BridgePlayer.EAST][suit] + "\n"
        for suit in range(NUM_SUITS - 1, -1, -1):
            rv += padding + cards[BridgePlayer.SOUTH][suit] + "\n"
        return rv

    def _format_vulnerability(self) -> str:
        if self._vulnerability[0]:
            vul_str = "All" if self._vulnerability[1] else "N/S"
        else:
            vul_str = "E/W" if self._vulnerability[1] else "None"
        return "Vul: " + vul_str + "\n"

    def _format_auction(self, trailing_query: bool) -> str:
        history_length = len(self._full_history)
        assert_lt(NUM_CARDS, history_length)
        rv = "\nWest  North East  South\n      "
        for i in range(NUM_CARDS, history_length):
            # add feed line
            if i % NUM_PLAYERS == NUM_PLAYERS - 1:
                rv += "\n"
            bid_string = get_bid_string(self._full_history[i].action - BIDDING_ACTION_BASE)
            rv += f"{bid_string:6}"

        if trailing_query:
            if history_length % NUM_PLAYERS == NUM_PLAYERS - 1:
                rv += "\n"
            rv += "?"
        return rv

    def _format_result(self):
        assert self.terminated()
        # not passed out, show declarer tricks
        rv = ""
        if self._contract.level:
            rv += f"\n\nDeclarer tricks: {self._num_declarer_tricks}"
        # show scores
        rv += f"\nScores: N/S {self._returns[BridgePlayer.NORTH]} E/W {self._returns[BridgePlayer.EAST]}"
        return rv

    def __repr__(self):
        rv = self._format_vulnerability() + self._format_deal()
        if len(self._full_history) > NUM_CARDS:
            rv += self._format_auction(trailing_query=False)
        if self.terminated():
            rv += self._format_result()
        return rv

    @property
    def holder(self):
        return self._holder

    @property
    def vul(self):
        return self._vulnerability

    @property
    def current_player(self):
        return self._current_player

    @property
    def history(self) -> List[int]:
        ret = []
        for player_action in self._full_history:
            ret.append(player_action.action)
        return ret

    @property
    def current_phase(self):
        return self._phase

    @property
    def ddt(self):
        return self._ddt

    @property
    def returns(self):
        return self._returns


"""
the auction observation tensor is made up by
vulnerabilities (2 * 2)
and for each player:
    Did this player pass before the opening bid? (1)
    Did this player make each bid? (35)
    Did this player double each bid? (35)
    Did this player redouble each bid? (35)
current player's hand (52)
4 + 52 + 4 * (3 * 35 + 1) = 480
"""
AUCTION_TENSOR_SIZE = 480


def encode_observation_tensor(state: BridgeBiddingState) -> torch.Tensor:
    # observation_tensor
    observation_tensor = torch.zeros(480, dtype=torch.float)
    vulnerability = state.vul
    holder = state.holder
    history = state.history
    # current player
    current_player = state.current_player
    partnership = get_partnership(current_player)
    ptr = 0
    """
    vulnerability encoding takes 4 bits,
    the first 2 bits represents whether the first partnership(i.e. NS) is vulnerable,
    [0, 1] represent vulnerable and [1,0] represent non-vulnerable, 
    same for the next 2 bits represent second partnership(i.e. EW)
    """
    observation_tensor[ptr + vulnerability[partnership]] = 1
    ptr += NUM_VULNERABILITIES
    observation_tensor[ptr + vulnerability[1 - partnership]] = 1
    ptr += NUM_VULNERABILITIES
    last_bid = 0
    for i in range(NUM_CARDS, len(history)):
        this_call = history[i] - BIDDING_ACTION_BASE if history[i] >= BIDDING_ACTION_BASE else history[i]
        """
        relative bidder is defined by clockwise, in current player's perspectives
        if i=52 and player is 0(North), then relative bidder is 0
        if i=52 and player is 1(East), then relative bidder is 3(because start from East, North is the third player)
        if i=52 and player is 2(South), then relative bidder is 2
        if i=52 and player is 3(West), then relative bidder is 1
        if i=53 and player is 0(North), then relative bidder is 1
        """
        relative_bidder = (i + NUM_PLAYERS - current_player) % NUM_PLAYERS
        # 4 bits for opening pass
        if last_bid == 0 and this_call == Calls.PASS:
            observation_tensor[ptr + relative_bidder] = 1

        """
        This part of encoding tracks for each bid and each player, does the player make, double and redoubled.
        the structure is 12(4 * 3) for each bid, and each four bit represents the relative player, three blocks is
        ordered by make, double and redouble.
        For example, if current player is East, and the bid 2C is made by North,
        so the relative player will be 3, and the relative index will be 
        4 (opening pass) + 12 * 5 (2C is the fifth bid) + 3(relative bidder) = 67
        if current player is South, and the bid 2C is made by North, and East doubles, 
        the relative index will be 4 + 12 * 5 + 2 = 66 and 4 + 12 * 5 + 4 + 3 = 71.
        if current player is West, and South redoubles after last two bids,3 relative indices will be
        4 + 12 * 5 + 1 = 65, 4 + 12 * 5 + 4 + 2 = 70, 4 + 12 * 5 + 4 + 4 + 3 = 75
        """
        if this_call == Calls.DOUBLE:

            observation_tensor[
                ptr + NUM_PLAYERS + (last_bid - FIRST_BID) * NUM_PLAYERS * 3 + NUM_PLAYERS + relative_bidder] = 1
        elif this_call == Calls.REDOUBLE:

            observation_tensor[
                ptr + NUM_PLAYERS + (last_bid - FIRST_BID) * NUM_PLAYERS * 3 + NUM_PLAYERS * 2 + relative_bidder] = 1
        elif this_call != Calls.PASS:
            last_bid = this_call

            observation_tensor[
                ptr + NUM_PLAYERS + (last_bid - FIRST_BID) * NUM_PLAYERS * 3 + relative_bidder] = 1
    ptr += 424  # 4* (1 + 35)
    for i in range(NUM_CARDS):
        if holder[i] == current_player:
            observation_tensor[ptr + i] = 1
    return observation_tensor


def state_from_trajectory(trajectory: Vector, ddt: Optional[np.ndarray] = None) -> BridgeBiddingState:
    """Get a BridgeBiddingState from a trajectory

    Args:
        trajectory (np.ndarray): the action trajectory
        ddt (Optional[np.ndarray]): the double dummy table, if not provided, the state will calculate
            if the trajectory makes a deal

    Returns:
        BridgeBiddingState: A BridgeBiddingState

    """
    if isinstance(ddt, np.ndarray):
        state = BridgeBiddingState(ddt=ddt)
    else:
        state = BridgeBiddingState()
    for action in trajectory:
        state.apply_action(action)
    return state
