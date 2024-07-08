from abc import ABC, abstractmethod

from treys import Card

from dataclasses import dataclass

@dataclass
class Player:
    name: str
    stack: int # TODO: needed? maybe not



class PokerDetection(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def stack_size(self) -> int:
        pass

    @abstractmethod
    def middle_pot(self) -> int:
        pass

    @abstractmethod
    def total_pot(self) -> int:
        pass


    @abstractmethod
    def current_bet(self) -> int:
        pass

    @abstractmethod
    def min_bet(self) -> int:
        pass

    # TODO: May not be implemented.
    @abstractmethod
    def big_blinds(self) -> int:
        pass

    @abstractmethod
    def community_cards(self) -> list[Card]:
        pass

    @abstractmethod
    def hole_cards(self) -> list[Card]:
        pass
    
    @abstractmethod
    def table_players(self) -> list:
        pass

    @abstractmethod
    def active_players(self) -> list:
        pass



