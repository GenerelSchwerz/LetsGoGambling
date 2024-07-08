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


# pretty python enum
class PokerStages:
    UNKNOWN = -1
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class PokerEvents:
    NEW_HAND = 0
    NEW_STAGE = 1

    # PLAYER_BET = 2



class PokerEventHandler(EventEmitter):

    def __init__(self, detector: PokerDetection):
        self.detector = detector
        self.was_turn = False
        self.last_stage = PokerStages.PREFLOP
        self.last_hand: list[Card] = []


    def tick(self):
        """
            Main loop for detecting events in the game.

            TODO handle player actions when NOT sitting
        """
        
        community_cards = self.detector.community_cards()

        hole_cards = self.detector.hole_cards()

        current_stage = PokerStages.UNKNOWN

        if len(community_cards) == 0:
            current_stage = PokerStages.PREFLOP

        elif len(community_cards) == 3:
            current_stage = PokerStages.FLOP

        elif len(community_cards) == 4:
            current_stage = PokerStages.TURN

        elif len(community_cards) == 5:
            current_stage = PokerStages.RIVER


        if current_stage != self.last_stage:

            # check to see if the cards are the same

            # technically not accurate but very unlikely to occur in practice
            if self.last_hand != hole_cards and current_stage == PokerStages.PREFLOP:
                self.emit(PokerEvents.NEW_HAND, (hole_cards, self.detector.table_players()))
                self.last_hand = hole_cards

            self.last_stage = current_stage

    















