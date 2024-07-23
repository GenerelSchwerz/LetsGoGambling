from abc import ABC, abstractmethod
from typing import Any
from event_emitter import EventEmitter

from .pokerDetection import PokerDetection


from treys import Card


class PokerEvents:
    """
        Abstact class that provides a blueprint for handling events in a poker game.

        Events:
            - NEW_HAND
                Args:
                    - hand: list[Card]
                    - big_blind: int
                    - small_blind: int

            - NEW_STAGE
                Args:
                    - last_stage: int
                    - current_stage: int
            
            
            - OUR_TURN
                Args:
                    - hole_cards: list[Card]
                    - community_cards: list[Card]
                    - facing_bet: int
                    - mid_pot: int
                    - total_pot: int

            - TICK
                Args:
                    - stage: int
                    - current_hand: list[Card]
                    - community_cards: list[Card]
        
    """


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
       

        














