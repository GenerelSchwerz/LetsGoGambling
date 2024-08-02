from abc import ABC, abstractmethod
import time
from typing import Any
from event_emitter import EventEmitter

from .PSPokerDetect import PSPokerImgDetect

from ...abstract.pokerDetection import PokerDetection
from ...abstract.pokerEventHandler import PokerStages, PokerEvents, PokerEventHandler


from treys import Card

import cv2



class PSEventEmitter(PokerEventHandler):


    def __init__(self, detector: PSPokerImgDetect):
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

        self.emit(PokerEvents.NEW_HAND, hand, big_blind, small_blind)
        self.last_hand = hand


    def __emit_our_turn(self, img: cv2.typing.MatLike, hole_cards: list[Card], community_cards: list[Card], stage: int):
        mid_pot = self.detector.middle_pot(img)
        current_bets = self.detector.current_bets(img)

        if len(current_bets) == 0:
            total_pot = mid_pot
            facing_bet = 0
        else:        
            total_pot = sum(current_bets) + mid_pot
            facing_bet = max(current_bets)

        table_players = self.detector.table_players(img)
        active_players = list(filter(lambda x: bool(x[0].active), table_players))

        to_print = list(map(lambda x: x[0], table_players))
        to_print_active = list(map(lambda x: x[0], active_players))
        print("[OUR TURN] table:", to_print, "active:", to_print_active, "facing:", facing_bet, "total:", total_pot, "mid:", mid_pot)

        self.emit(PokerEvents.OUR_TURN,
                    hole_cards,
                    community_cards,
                    stage,
                    facing_bet,
                    self.detector.min_bet(img),
                    mid_pot,
                    total_pot,
                    self.detector.stack_size(img),
                    len(active_players) - 1  # exclude us
                  )
        # self.our_turn = True # slightly redundant code as this should only ever be called if this is supposed to be true.
        
        

    def tick(self, image: cv2.typing.MatLike):
        start = time.time()
        try:
            community_cards = self.detector.community_cards(image)
            current_hand = self.detector.hole_cards(image)
        except (KeyError, ValueError) as e:
            # OCR failed somewhere, so we're just going to skip this iteration for the time being.
            print(f"Error in detecting community cards: {e}")
            return


        # this is triplejack specific.

        if self.last_stage == PokerStages.SHOWDOWN or self.last_stage == PokerStages.RIVER:
            if (card_len := len(community_cards)) < 5 and card_len > 0:
                current_stage = PokerStages.SHOWDOWN # currently displaying winning cards
            elif card_len == 0:
                current_stage = PokerStages.PREFLOP
            else:
                current_stage = self.last_stage

        else:
            if (card_len := len(community_cards)) == 0:
                current_stage = PokerStages.PREFLOP
        
            elif card_len <= 3: # transitioning to flop
                current_stage = PokerStages.FLOP

            elif card_len == 4:
                if self.last_stage == PokerStages.RIVER:
                    current_stage = PokerStages.PREFLOP # new hand
                else:
                    current_stage = PokerStages.TURN

            elif card_len == 5:
                current_stage = PokerStages.RIVER

            else:
                # we're in the showdown stage, other cards above peoples' heads are visible. We're waiting for hand to end, so just end early.
                self.last_stage = PokerStages.SHOWDOWN
                return
                # raise ValueError(f"Invalid number of community cards, found {len(community_cards)}")

        # current_hand = None
    
        if current_stage != self.last_stage:
            self.emit(PokerEvents.NEW_STAGE, self.last_stage, current_stage)

            if current_stage == PokerStages.PREFLOP:
                if (current_hand != self.last_hand or len(current_hand)== 0):
                    self.__emit_new_hand(image, current_hand)
             

        elif current_stage == PokerStages.PREFLOP:
            # current_hand = self.detector.hole_cards(image)

            if current_hand != self.last_hand and (hand_len := len(current_hand)) == 2:
                if hand_len < 2:
                    # shouldn't happen.
                    raise ValueError(f"Invalid number of hole cards (too little), found {hand_len}")
                
                self.__emit_new_hand(image, current_hand)
            


        if current_hand is None:
            current_hand = self.last_hand


        # =========================
        # Now, check for our turn.
        # =========================
            

        if len(current_hand) == 0:
            # we're not in a hand, so we're not going to check for our turn.
            self.our_turn = False
            self.last_stage = current_stage
            self.emit(PokerEvents.TICK, current_stage, current_hand, community_cards)
            return


        check_button = self.detector.check_button(image, threshold=0.8)
        if check_button is not None:
            our_turn = True
        else:
            fold_button = self.detector.fold_button(image, threshold=0.8)
            if fold_button is not None:
                our_turn = True
                    
            else:
                our_turn = False

        if our_turn != self.our_turn and our_turn:
            self.__emit_our_turn(image, current_hand, community_cards, current_stage)

        elif our_turn == self.our_turn and our_turn:
            # check if stage has changed
            if current_stage > self.last_stage:
                self.__emit_our_turn(image, current_hand, community_cards, current_stage)
        
        self.our_turn = our_turn
        self.last_stage = current_stage
        self.emit(PokerEvents.TICK, current_stage, current_hand, community_cards)
        














