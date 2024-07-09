from typing import Any
from ...abstract.impl.eventEmitter import PokerStages


def suit_full_name_to_abbrev(suit_full_name: str) -> str:
    if suit_full_name == "hearts":
        return "h"
    elif suit_full_name == "diamonds":
        return "d"
    elif suit_full_name == "clubs":
        return "c"
    elif suit_full_name == "spades":
        return "s"
    
    raise ValueError("Invalid suit name")
    

def card_to_abbrev(card: str) -> str:
    if card == "10":    
        return "T"
    return card


def pretty_str_to_int(str: str) -> int:

    str = str.replace(",", "").lower()

    if str[-1] == "k":
        return int(float(str[:-1]) * 1000)
    
    if str[-1] == "m":
        return int(float(str[:-1]) * 1000000)
    
    return int(str)


def cards_to_stage(cards: list[Any]) -> PokerStages:
    if len(cards) == 0:
        return PokerStages.PREFLOP
    elif len(cards) <= 3:
        return PokerStages.FLOP
    elif len(cards) == 4:
        return PokerStages.TURN
    else:
        return PokerStages.RIVER