from abc import ABC, abstractmethod

from treys import Card

from pokerbot.abstract.pokerDetection import Player

from .pokerEventHandler import PokerEvents, PokerStages, PokerEventHandler


class PokerDecisionChoice:
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3

    def __init__(self, choice: int, amount: int = 0):
        self.choice = choice
        self.amount = amount

    @staticmethod
    def fold():
        return PokerDecisionChoice(PokerDecisionChoice.FOLD)
    
    @staticmethod
    def check():
        return PokerDecisionChoice(PokerDecisionChoice.CHECK)
    
    @staticmethod
    def call():
        return PokerDecisionChoice(PokerDecisionChoice.CALL)
    
    @staticmethod
    def bet(amount: int):
        return PokerDecisionChoice(PokerDecisionChoice.BET, amount)
    



class PokerDecisionMaking(ABC):
    

    def on_turn(self,
                hole_cards: list[Card],
                community_cards: list[Card],
                stage: int,
                facing_bet: int,
                min_bet: int,
                mid_pot: int,
                total_pot: int,
                big_blind: int,
                stack_size: int,
                active_opponents: int,
                ) -> PokerDecisionChoice:
        pass

    def on_new_hand(
            self,
            hole_cards: list[Card],
            big_blind: int,
            small_blind: int,
            players: list[Player],
            bets: dict[Player, float]
            ):
        pass
    