from typing import Union


import cv2

from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement

from treys import Card

from logging import Logger

from ..abstract.pokerInteract import PokerInteract

from .utils import prepare_ss
from .base.TJPokerDetect import TJPokerDetect, report_info


from ..abstract.pokerEventHandler import PokerEvents, PokerStages
from .base.gameEvents import TJEventEmitter

import time

from .TJInteract import TJInteract, TJPage

from ..abstract.impl.full_client import AClient


class TJClient:

    currently_running = 0

    """
    This class is the main class for the poker bot.
    """

    def __init__(self, headless=False, debug=False, **kwargs):
        self.debug = debug

        self.detector = TJPokerDetect()
        self.event_handler = TJEventEmitter(self.detector)

        self.headless = headless
        self.interactor: PokerInteract = None

        self.ss = None

    def handle_our_turn(self, last_hand: list[Card], facing_bet: int, mid_pot: int, total_pot: int):
        print(f"[OUR TURN] hand: {Card.ints_to_pretty_str(last_hand)} facing: {facing_bet} mid: {mid_pot} total: {total_pot}")

        # logic can go here
        call = self.interactor.call(self.ss)
        if not call:
            print("call failed, trying checking")
            check = self.interactor.check(self.ss)
            if not check:
                print("both call and check failed")
            else:
                print("Call failed, but check succeeded")
        else:
            print("Call succeeded")
            


    def handle_new_hand(self, hand: list[Card], big_blind: int, small_blind: int):
        print(f"[HAND] {Card.ints_to_pretty_str(hand)} BB: {big_blind} SB: {small_blind}")

    def handle_new_stage(self, last_stage: int, current_stage: int):
        print(f"[STAGE] last: {PokerStages.to_str(last_stage)} current: {PokerStages.to_str(current_stage)}")

    def handle_tick(self, stage: int, current_hand: list[Card], community_cards: list[Card]):
        return
        print(f"[TICK] hole: {Card.ints_to_pretty_str(current_hand)} community: {Card.ints_to_pretty_str(community_cards)} stage: {PokerStages.to_str(stage)}")


    def run_ticks(self):
        time_split = 2
        last_time = time.time()
        while True:

            self.ss = self.interactor._ss()
            self.event_handler.tick(self.ss)
            now = time.time()
            dur = max(0, time_split - (now - last_time))
            print("Sleeping for", round(dur * 1000) / 1000, "seconds. Been", round((now - last_time) * 1000) / 1000, "seconds taken to calculate since last tick.",round(( dur + now - last_time) * 1000) / 1000, "seconds total")
            time.sleep(dur)
            last_time = time.time()

    def start(self):
        if self.interactor is not None:
            self.interactor.shutdown()

        if not self.detector.is_initialized():
            self.detector.load_images()

        self.interactor = TJInteract(headless=self.headless, detector=self.detector)

        self.interactor.start("ForTheChips", "WooHoo123!")

        self.event_handler.on(PokerEvents.OUR_TURN, self.handle_our_turn)
        self.event_handler.on(PokerEvents.NEW_HAND, self.handle_new_hand)
        self.event_handler.on(PokerEvents.NEW_STAGE, self.handle_new_stage)
        self.event_handler.on(PokerEvents.TICK, self.handle_tick)

        # TODO: better tick code (potentially offload to another thread)

        self.run_ticks()



    def stop(self):
        if self.interactor is not None:
            self.interactor.shutdown()
            self.interactor = None

        self.event_handler.remove(PokerEvents.OUR_TURN, self.handle_our_turn)
        self.event_handler.remove(PokerEvents.NEW_HAND, self.handle_new_hand)
        self.event_handler.remove(PokerEvents.NEW_STAGE, self.handle_new_stage)
        self.event_handler.remove(PokerEvents.TICK, self.handle_tick)


def run_ticks(_bot):
    time_split = 2
    last_time = time.time()
    while True:

        _bot.ss = _bot.interactor._ss()
        _bot.event_handler.tick(_bot.ss)
        now = time.time()
        dur = max(0, time_split - (now - last_time))
        print("Sleeping for", round(dur * 1000) / 1000, "seconds. Been", round((now - last_time) * 1000) / 1000, "seconds taken to calculate since last tick.",round(( dur + now - last_time) * 1000) / 1000, "seconds total")
        time.sleep(dur)
        last_time = time.time()

def main():

    from ..all.simpleLogic import SimpleDecisions

    detector = TJPokerDetect()
    detector.load_images()

    event_handler = TJEventEmitter(detector=detector)
    interactor = TJInteract(headless=False, detector=detector)

    logic = SimpleDecisions()

    bot = AClient(
        event_handler=event_handler,
        detector=detector,
        interactor=interactor,
        logic=logic,
        debug=True
    )



    try:
        bot.start("ForTheChips", "WooHoo123!")

        run_ticks(bot)
    except KeyboardInterrupt:
        print("bot finished")
        bot.stop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        bot.stop()


if __name__ == "__main__":
    main()
