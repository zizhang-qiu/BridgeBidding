#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:bridge_vars.py
@time:2023/02/16
"""
from enum import IntEnum


class Denomination(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    NO_TRUMP = 4


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class DoubleStatus(IntEnum):
    UNDOUBLED = 1
    DOUBLED = 2
    REDOUBLED = 4


NUM_DENOMINATIONS = 5
NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_CARDS = 52
NUM_CARDS_PER_SUIT = 13
NUM_PARTNERSHIPS = 2  # NS and EW
NUM_BID_LEVELS = 7  # 1C-7C
NUM_VULNERABILITIES = 2  # vul and non-vul
NUM_BIDS = 35  # 1C, 1D ... 7S, 7NT
NUM_OTHER_CALLS = 3  # pass, double, redouble
NUM_CALLS = 38
NUM_CARDS_PER_HAND = 13  # 13 cards for each player
NUM_TRICKS = 13  # 13 tricks at most

NUM_DOUBLE_STATES = 3
NUM_CONTRACTS = 421  # 1 + (35 * 3 * 4)

# level and denomination are for bid string
LEVEL_STR = "-1234567"
DENOMINATION_STR = "CDHSN"
DENOMINATIONS = ["C","D","H","S"]

# suit and rank are for card string
SUIT_STR = "CDHS"
RANK_STR = "23456789TJQKA"

PLAYER_STR = "NESW"

# scores
MAX_SCORE = 7600
MAX_IMP = 24

# used for pbn
PBN_PREFIX = """% PBN 2.1
% EXPORT
%Content-type: text/x-pbn; charset=ISO-8859-1
%Creator: BridgeComposer Version 5.98
%Created: Fri Feb 17 09:54:38 2023 +0800
%BCOptions Center STBorder STShade
%BidAndCardSpacing Thin
%BoardsPerPage 1
%CardTableColors #008000,#ffffff,#aaaaaa
%DDAFormat 0
%DefaultPagination 0
%EventSpacing 0
%Font:CardTable "Arial",11,400,0
%Font:Commentary "Times New Roman",12,400,0
%Font:Diagram "Times New Roman",12,400,0
%Font:Event "Times New Roman",12,400,0
%Font:FixedPitch "Courier New",10,400,0
%Font:HandRecord "Arial",11,400,0
%GutterSize 500,500
%HRTitleDate 0
%HRTitleEvent ""
%HRTitleSetID ""
%HRTitleSetIDPrefix ""
%HRTitleSite ""
%HtmlClubs entity,"https://bridgecomposer.com/suitimg/c.gif"
%HtmlDiamonds entity,"https://bridgecomposer.com/suitimg/d.gif"
%HtmlHearts entity,"https://bridgecomposer.com/suitimg/h.gif"
%HtmlNavBar 0.75,#cfe2f3
%HtmlSpades entity,"https://bridgecomposer.com/suitimg/s.gif"
%Margins 1000,1000,1000,1000
%PaperSize 1,0,0,2
%ParaIndent 0
%PipColors #000000,#ff0000,#ff0000,#000000
%PipFont "Symbol","Symbol",2,0xAA,0xA9,0xA8,0xA7
%ScoreTableColors #e6e6e6,#000000
%SelectedBoard 1
%ShowBoardLabels 2
%ShowCardTable 2
%TSTCustomSortOrder Default
%TSTReport List
%TSTReportOrder ByNumber
%TSTReportShade Yes"""
PBN_TEMPLATE = """[Event ""]
[Site ""]
[Date ""]
[Board ""]
[West ""]
[North ""]
[East ""]
[South ""]
[Dealer "{dealer}"]
[Vulnerable "None"]
[Deal "{deal}"]
[Scoring ""]
[Declarer ""]
[Contract ""]
[Result ""]
[BCFlags "1f"]"""

PLUS_MINUS_SYMBOL = "\u00B1"
SECONDS_PER_MINUTE = 60

if __name__ == '__main__':
    print(PBN_PREFIX + PBN_TEMPLATE.format(dealer="N", deal=""))
