from treys import Card

from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice

import random

# hello

# everything is now set to make decisions


class SimpleDecisions(PokerDecisionMaking):

    def on_turn(self,
                hole_cards: list[Card],
                community_cards: list[Card],
                facing_bet: int,
                min_bet: int,
                mid_pot: int,
                total_pot: int,
                big_blind: int,
                stack_size: int,
                active_opponents: int,
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
