from abc import ABC, abstractmethod
from typing import Any
from eventemitter import EventEmitter

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
    NEW_HAND = "NewHand"
    NEW_STAGE = "NewStage"


class PokerStages:
    UNKNOWN = -1
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4



class PokerEventHandler(EventEmitter):


    def __init__(self, detector: PokerDetection):
        super().__init__()
        self.detector = detector
        
        self.last_stage = PokerStages.UNKNOWN
        self.last_hand: list[Card] = []

    
    def tick(self):

        community_cards = self.detector.community_cards()

        if len(community_cards) == 0:
            current_stage = PokerStages.PREFLOP
        
        elif len(community_cards) == 3:
            current_stage = PokerStages.FLOP

        elif len(community_cards) == 4:
            current_stage = PokerStages.TURN

        elif len(community_cards) == 5:
            current_stage = PokerStages.RIVER

        else:
            raise ValueError(f"Invalid number of community cards, found {len(community_cards)}")

        if current_stage != self.last_stage:
            current_hand = self.detector.hole_cards()

            if current_hand != self.last_hand and current_stage == PokerStages.PREFLOP:
                self.emit(PokerEvents.NEW_HAND, (current_hand, self.detector.players_at_table()))
                self.last_stage = current_stage
                self.last_hand = current_hand

        elif current_stage == PokerStages.PREFLOP:
            if current_hand != self.last_hand:
                self.emit(PokerEvents.NEW_HAND, (current_hand, self.detector.players_at_table()))
                self.last_stage = current_stage
                self.last_hand = current_hand
            



        














