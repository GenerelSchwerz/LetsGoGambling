from abc import ABC, abstractmethod

from treys import Card

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
    
    @abstractmethod
    def on_turn(self, cards: list[Card], facing_bet: int, mid_pot: int, total_pot: int) -> PokerDecisionChoice:
        pass
