
import random
import subprocess
from typing import Union

from ...all.windows import AWindowManager, UnixWindowManager, WindowsWindowManager, get_all_windows_matching
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

from logging import Logger

log = Logger("PokerBot")


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

    def start(self, *args):
        pass


    def shutdown(self):
        if self.wm is not None:
            self.wm.close()

        else:
            raise RuntimeError("Window manager not initialized")


    def click(self, x: int, y: int, amt=1):
        dims = self.wm.get_window_dimensions()
        x1 = x + dims[0]
        y1 = y + dims[2]

        print(f"Clicking at {x}, {y}")
        print(pyautogui.position())
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

        clicks = int((sb * round((amt - self.detector.min_bet(ss)) / sb)) / sb)
        if clicks > 20:
            clicks = int(clicks * (0.9 + (0.2 * random.random())))

        press_plus = self.detector.plus_button(ss, threshold=0.8)
        if press_plus is None:
            print("Could not find plus button")
            return False


        self.click(*mid(press_plus), clicks)
        
        button = self.detector.bet_button(ss, threshold=0.8)
        if button is None:
            button = self.detector.raise_button(ss, threshold=0.8)
            if button is None:
                button = self.detector.allin_button(ss, threshold=0.8)
                if button is None:
                    return False

        self.click(*mid(button))
        return True
    

    def reraise(self, amt: int, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.raise_button(ss, threshold=0.8)

        if button is None:
            return False

        self.click(*mid(button))
        return True

    def check(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False

        if ss is None:
            ss = self._ss()

        button = self.detector.check_button(ss, threshold=0.8)

        if button is None:
            return False

        self.click(*mid(button))
        return True

    def fold(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.fold_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.click(*mid(button))
        return True

    def call(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False


        if ss is None:
            ss = self._ss()


        button = self.detector.call_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.click(*mid(button))
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