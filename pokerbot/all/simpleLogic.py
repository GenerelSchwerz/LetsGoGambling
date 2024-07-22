from treys import Card

from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice

class SimpleDecisions(PokerDecisionMaking):
    
    def on_turn(self, cards: list[Card], facing_bet: int, mid_pot: int, total_pot: int) -> PokerDecisionChoice:
        return PokerDecisionChoice.fold()