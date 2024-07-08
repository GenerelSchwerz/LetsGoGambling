from abc import ABC, abstractmethod
from typing import Any
from eventemitter import EventEmitter



"""
    Abstact class that provides a blueprint for handling events in a poker game.

    Events:
        - NewHand: Game has started, we've been dealt cards
            - (cards, players at table)
        - MyTurn: It's our turn to act
            - (current bet, pot, players in hand)
        - PlayerBets: A player acts 
            - (player, action, amount?)

        # TODO: good enough for now
    
"""

class PokerEventHandler(ABC, EventEmitter):


    @abstractmethod
    def fetch_game_instance(self) -> Any:
        pass















