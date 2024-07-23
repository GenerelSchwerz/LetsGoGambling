from treys import Card

from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice

import random

# hello

# everything is now set to make decisions


import math
import random



class AlgoDecisionMode:
    HOLDEM = 0
    OMAHA = 1



class AlgoDecisions(PokerDecisionMaking):


    def __init__(self, mode=int):
        self.mode = mode

    def calculate_equity(self, hole_cards: list[Card], board_cards: list[Card], num_opponents) -> float:
        return 0.0


    def on_turn(
        self, cards: list[Card], community_cards: list[Card], facing_bet: int, mid_pot: int, total_pot: int
    ) -> PokerDecisionChoice:
        choice = random.randrange(0, 4)

        if choice == 0:
            return PokerDecisionChoice.fold()
        elif choice == 1:
            return PokerDecisionChoice.check()
        elif choice == 2:
            return PokerDecisionChoice.call()
        else:
            return PokerDecisionChoice.bet(random.randrange(0, 100))
