from abc import ABC, abstractmethod
from typing import Union

from treys import Card

from dataclasses import dataclass

@dataclass
class Player:
    name: str
    stack: int # TODO: needed? maybe not
    active: Union[bool, None] = None

    def __key(self):
        return (self.name, self.stack, self.active)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Player):
            return self.__key() == other.__key()
        return NotImplemented

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
    def current_bets(self) -> list[int]:
        pass

    @abstractmethod
    def current_bet(self) -> int:   
        pass

    @abstractmethod
    def min_bet(self) -> int:
        pass

    @abstractmethod
    def big_blind(self) -> int:
        pass

    @abstractmethod
    def small_blind(self) -> int:
        pass

    @abstractmethod
    def community_cards(self) -> list[Card]:
        pass

    @abstractmethod
    def community_cards_and_locs(self) -> list[tuple[Card, tuple[int, int]]]:
        pass

    @abstractmethod
    def hole_cards(self) -> list[Card]:
        pass

    @abstractmethod
    def hole_cards_and_locs(self) -> list[tuple[Card, tuple[int, int]]]:
        pass
    
    @abstractmethod
    def table_players(self) -> list:
        pass

    @abstractmethod
    def active_players(self) -> list:
        pass



