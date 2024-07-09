from abc import ABC, abstractmethod
from typing import Any
from eventemitter import EventEmitter

from ..pokerDetection import PokerDetection


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
    NEW_HAND = "NewHand"
    NEW_STAGE = "NewStage"
    TEST = "Test"


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
    



class PokerEventHandler(EventEmitter):


    def __init__(self, detector: PokerDetection):
        super().__init__()
        self.detector = detector
        
        self.last_stage = PokerStages.UNKNOWN
        self.last_hand: list[Card] = []

    
    def tick(self, *args):

        community_cards = self.detector.community_cards(*args)

        if len(community_cards) == 0:
            current_stage = PokerStages.PREFLOP
        
        elif len(community_cards) <= 3: # transitioning to flop
            current_stage = PokerStages.FLOP

        elif len(community_cards) == 4:
            current_stage = PokerStages.TURN

        elif len(community_cards) == 5:
            current_stage = PokerStages.RIVER

        else:
            raise ValueError(f"Invalid number of community cards, found {len(community_cards)}")

   
        print(PokerStages.to_str(current_stage), PokerStages.to_str(self.last_stage))
        
        current_hand = self.detector.hole_cards(*args)

        print(current_hand, self.last_hand)
        print(community_cards)

        if current_stage != self.last_stage:
            current_hand = self.detector.hole_cards(*args)

            if current_hand != self.last_hand and current_stage == PokerStages.PREFLOP:
                self.emit(PokerEvents.NEW_HAND, (current_hand))
                self.last_hand = current_hand

        elif current_stage == PokerStages.PREFLOP:
            current_hand = self.detector.hole_cards(*args)

            if current_hand != self.last_hand:
                self.emit(PokerEvents.NEW_HAND, (current_hand))
                self.last_hand = current_hand
            
        self.last_stage = current_stage

        self.emit(PokerEvents.TEST, (current_stage, community_cards))



        














