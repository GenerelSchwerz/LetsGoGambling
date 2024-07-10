from abc import ABC, abstractmethod
from typing import Any
from event_emitter import EventEmitter

from .pokerDetection import PokerDetection


from treys import Card

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

class PokerEvents:
    TEST = -1
    NEW_HAND = 0
    NEW_STAGE = 1

    OUR_TURN = 2

    TICK = 3
  

    pretty_str = {
        TEST: "Test",
        NEW_HAND: "New Hand",
        NEW_STAGE: "New Stage",
     
        OUR_TURN: "Our Turn",
        TICK: "Info"
    }

    def to_str(event: str) -> str:
        return PokerEvents.pretty_str[event]


class PokerStages:
    UNKNOWN = -1
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

    pretty_str = {
        UNKNOWN: "Unknown",
        PREFLOP: "Preflop",
        FLOP: "Flop",
        TURN: "Turn",
        RIVER: "River",
        SHOWDOWN: "Showdown"
    
    }

    @staticmethod
    def to_str(stage: int) -> str:
        return PokerStages.pretty_str[stage]
    



class PokerEventHandler(ABC, EventEmitter):
    
    @abstractmethod
    def tick(self, *args):
        pass
       

        














