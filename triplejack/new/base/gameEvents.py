from abc import ABC, abstractmethod
from typing import Any
from event_emitter import EventEmitter

from .imgDetect import TJPokerDetect

from ...abstract.pokerDetection import PokerDetection
from ...abstract.pokerEventHandler import PokerStages, PokerEvents, PokerEventHandler


from treys import Card

import cv2

"""
    Abstact class that provides a blueprint for handling events in a poker game.

    Events:
        - NewHand: Game has started, we've been dealt cards
            - (cards, players at table)
        - MyTurn: It's our turn to act
            - (current bet, pot, players in hand)
        - PlayerBets: A player acts 
            - (player, action, amount?)    
"""





class TJEventEmitter(PokerEventHandler):


    def __init__(self, detector: TJPokerDetect):
        super().__init__()
        self.detector = detector
        
        self.last_stage = PokerStages.PREFLOP
        self.last_hand: list[Card] = []

        self.sitting = False
        self.our_turn = False

    
    def __emit_new_hand(self, img: cv2.typing.MatLike, hand: list[Card]):

        # this occasionally fails.
        small_blind = self.detector.small_blind(img)

        # eager, this is most likely not necessary.
        big_blind = self.detector.big_blind(img) 

        print(big_blind, small_blind)

        self.emit(PokerEvents.NEW_HAND, hand, big_blind, small_blind)
        self.last_hand = hand


    def __emit_our_turn(self, img: cv2.typing.MatLike):
        mid_pot = self.detector.middle_pot(img)
        current_bets = self.detector.current_bets(img)

        if len(current_bets) == 0:
            total_pot = mid_pot
            facing_bet = 0
        else:        
            total_pot = sum(current_bets) + mid_pot
            facing_bet = max(current_bets)

        self.emit(PokerEvents.OUR_TURN, self.last_hand, facing_bet, mid_pot, total_pot)
        # self.our_turn = True # slightly redundant code as this should only ever be called if this is supposed to be true.
        
        

    def tick(self, image: cv2.typing.MatLike):

        try:
            community_cards = self.detector.community_cards(image)
        except (KeyError, ValueError) as e:
            # OCR failed somewhere, so we're just going to skip this iteration for the time being.
            print(f"Error in detecting community cards: {e}")
            return


        # this is triplejack specific.

        if self.last_stage == PokerStages.SHOWDOWN or self.last_stage == PokerStages.RIVER:
            if (test := len(community_cards)) < 5 and test > 0:
                current_stage = PokerStages.SHOWDOWN # currently displaying winning cards
            elif test == 0:
                current_stage = PokerStages.PREFLOP
            else:

                # we are detecting false positives (players' displayed cards) as community cards.
                # TODO ideal fix is cropping the search range to just the table. Didn't do that yet.
                current_stage = self.last_stage

        else:
            if len(community_cards) == 0:
                current_stage = PokerStages.PREFLOP
        
            elif len(community_cards) <= 3: # transitioning to flop
                current_stage = PokerStages.FLOP

            elif len(community_cards) == 4:
                if self.last_stage == PokerStages.RIVER:
                    current_stage = PokerStages.PREFLOP # new hand
                else:
                    current_stage = PokerStages.TURN

            elif len(community_cards) == 5:
                current_stage = PokerStages.RIVER

            else:
                # we're in the showdown stage, other cards above peoples' heads are visible. We're waiting for hand to end, so just end early.
                self.last_stage = PokerStages.SHOWDOWN
                return
                # raise ValueError(f"Invalid number of community cards, found {len(community_cards)}")

        current_hand = None
    
        if current_stage != self.last_stage:
            current_hand = self.detector.hole_cards(image)
        
            self.emit(PokerEvents.NEW_STAGE, self.last_stage, current_stage)

            if (current_hand != self.last_hand or len(current_hand)== 0) and current_stage == PokerStages.PREFLOP:
                self.__emit_new_hand(image, current_hand)
             

        elif current_stage == PokerStages.PREFLOP:
            current_hand = self.detector.hole_cards(image)

            if current_hand != self.last_hand and (hand_len := len(current_hand)) == 2:
                if hand_len < 2:
                    # shouldn't happen.
                    raise ValueError(f"Invalid number of hole cards, found {hand_len}")
                
                self.__emit_new_hand(image, current_hand)
            
        self.last_stage = current_stage

        if current_hand is None:
            current_hand = self.last_hand


        # =========================
        # Now, check for our turn.
        # =========================
        
        our_turn = False
        call_button = self.detector.call_button(image)
        if call_button is not None:
            our_turn = True
        else:
            check_button = self.detector.check_button(image)
            if check_button is not None:
                our_turn = True
                    
            else:
                our_turn = False

        if our_turn != self.our_turn and our_turn == True:
            self.__emit_our_turn(image)
        
        self.our_turn = our_turn
        
        
        self.emit(PokerEvents.TICK, current_stage, current_hand, community_cards)



        














