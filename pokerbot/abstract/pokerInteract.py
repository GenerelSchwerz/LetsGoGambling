from abc import ABC, abstractmethod


"""
    Abstract class meant to generalize interacting with sites in-game.

    For example, this class provides a blueprint of "betting", "folding", "checking", etc.
"""
class PokerInteract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def bet(self, amt: int, sb: int, bb: int) -> bool:
        pass

    # TODO: Perhaps merge into bet (with a flag for reraise)
    @abstractmethod
    def reraise(self, amt: int) -> bool:
        pass

    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def fold(self) -> bool:
        pass

    @abstractmethod
    def call(self) -> bool:
        pass

    # TODO: may not be implemented (no need)
    @abstractmethod
    def sit(self) -> bool:
        pass
    
    @abstractmethod
    def shutdown(self):
        pass








