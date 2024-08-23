
import random
import subprocess
from typing import Union

from ...all.windows import *
from ...abstract.pokerInteract import PokerInteract
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement

from .CPokerDetect import CPokerImgDetect

import pyautogui

import time
import os

import cv2

import logging

log = logging.getLogger(__name__)


def mid(tupl: tuple[int, int, int, int]) -> tuple[int, int]:
    return (tupl[0] + tupl[2]) // 2, (tupl[1] + tupl[3]) // 2

class CPokerInteract(PokerInteract):
    """
    TODO MAJOR! Support multiple tables.
    """

    def __init__(self, detector: CPokerImgDetect, wm: AWindowManager):
        super().__init__()
        

        # performance in case multiple instances
        
        self.detector = detector
        if not self.detector.is_initialized():
            raise ValueError("Detector not initialized")
    
        self.wm = wm


    def shutdown(self):
        if self.wm is not None:
            self.wm.close()

        else:
            raise RuntimeError("Window manager not initialized")


    def click(
        self,
        x: int,
        y: int,
        amt=1,
        bb: Union[tuple[int, int, int, int], None] = None,
        bias=0.7,
        travel_time: Union[int, None] = None,
    ):
        dims = self.wm.get_window_dimensions()

        log.debug(f"org x, y wanted for clicks: {x + dims[0]}, {y + dims[2]}")

        # randomly offset x and y within bb if present, bias towards current x and y
        if bb is not None:
            x += int((bb[2] - bb[0]) * bias * random.random())
            y += int((bb[3] - bb[1]) * bias * random.random())

        x1 = x + dims[0]
        y1 = y + dims[2]

        # randomly select travel time if not provided
        # bias towards longer traveling if x1, y1 is far from current position
        # far being determined by the distance between the two points relative to dims [x, x1, y, y1]
        if travel_time is None:
            cur_x, cur_y = pyautogui.position()
            travel_time = 0.2 + (0.1 * random.random())
            if (abs(cur_x - x1) / dims[1]) + (abs(cur_y - y1) / dims[3]) > 0.5:
                travel_time = 0.2 + (0.2 * random.random())

        log.debug(f"Clicking at {x1}, {y1}")
        pyautogui.moveTo(x1, y1, duration=travel_time, tween=pyautogui.easeInOutQuad)
        pyautogui.click(x1, y1, clicks=amt, interval=0.05)



    def _ss(self):  # why not make this a class property/field that is updated once per tick? - amelia
        ss = self.wm.ss()

        cv2.imwrite(f"./midrun/{int(time.time() * 100_000) / 100_000}.png", ss)
        return ss
        
    

    def bet(self, amt: int, sb: int, bb: int, ss=None) -> bool: 
        # if self.page != TJPage.TABLE:
        #     return False
        if ss is None:
            ss = self._ss()

        min_bet = self.detector.min_bet(ss)

        # clicks increment by big blind. So starting amt is min_bet, then add X big blinds to get to amt
        clicks = 1
        while min_bet + (clicks * bb) < amt:
            clicks += 1

        # format string
        log.debug(f"Clicking {clicks} times {min_bet + (clicks * bb)} of {amt}")

        if clicks > 20:
            clicks = int(clicks * (0.9 + (0.2 * random.random())))
            log.debug(f"Adjusted clicks to {clicks}")

        press_plus = self.detector.plus_button(ss, threshold=0.8)
        if press_plus is None:
            log.debug("Could not find plus button")
            return False

        self.click(*mid(press_plus), clicks, bb=press_plus)

        button = self.detector.bet_button(ss, threshold=0.8)
        if button is None:
            button = self.detector.raise_button(ss, threshold=0.8)
            if button is None:
                button = self.detector.allin_button(ss, threshold=0.8)
                if button is None:
                    return False

        self.click(*mid(button), bb=button)
        return True
    

    def reraise(self, amt: int, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.raise_button(ss, threshold=0.8)

        if button is None:
            return False

        self.click(*mid(button), bb=button)
        return True

    def check(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False

        if ss is None:
            ss = self._ss()

        button = self.detector.check_button(ss, threshold=0.8)

        if button is None:
            return False

        self.click(*mid(button), bb=button)
        return True

    def fold(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.fold_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.click(*mid(button), bb=button)
        return True

    def call(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False


        if ss is None:
            ss = self._ss()


        button = self.detector.call_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.click(*mid(button), bb=button)
        return True

    def sit(self) -> bool:
        pass



if __name__ == "__main__":
    detector = CPokerImgDetect()
    detector.load_images()
    from ...all.windows import UnixWindowManager
    test = CPokerInteract(detector=detector, path_to_exec="/home/generel/.local/share/applications/wine/Programs/PokerStars.net/PokerStars.net.desktop")
    test.open_pokerstars()

    # time.sleep(10)
    # test.login("GenerelSchwerz", "Rocky1928!")

    test.wait_until_in_room()

    try:
        time.sleep(600)
    except KeyboardInterrupt:
        pass
    finally:
        test.shutdown()