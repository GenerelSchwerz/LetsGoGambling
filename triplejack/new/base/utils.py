from typing import Any
from ...abstract.pokerEventHandler import PokerStages


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
    number = str.lower()
    # remove comma/period
    has_period = '.' in number or ',' in number
    new_number = number.replace(',', '')
    new_number = new_number.replace('.', '')
    new_number = new_number.replace('k', '00' if has_period else '000')
    new_number = new_number.replace('m', '00000' if has_period else '000000')
    try:
        return int(new_number)
    except ValueError:
        print(f'Could not convert number to int: {number} ({new_number})')
        return 0


def cards_to_stage(cards: list[Any]) -> PokerStages:
    if len(cards) == 0:
        return PokerStages.PREFLOP
    elif len(cards) <= 3:
        return PokerStages.FLOP
    elif len(cards) == 4:
        return PokerStages.TURN
    else:
        return PokerStages.RIVER