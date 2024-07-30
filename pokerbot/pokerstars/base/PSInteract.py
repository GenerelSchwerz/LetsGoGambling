
import random
import subprocess
from typing import Union

from ...all.windows import AWindowManager, get_all_windows_matching
from ...abstract.pokerInteract import PokerInteract
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement

from .PSPokerDetect import PSPokerDetect

import pyautogui

import time
import os
from logging import Logger

log = Logger("PokerBot")


def mid(tupl: tuple[int, int, int, int]) -> tuple[int, int]:
    return (tupl[0] + tupl[2]) // 2, (tupl[1] + tupl[3]) // 2

class PSInteract(PokerInteract):

    def __init__(self, detector: PSPokerDetect, path_to_exec: str):
        super().__init__()
        

        # performance in case multiple instances
        if detector:
            self.detector = detector
            if not self.detector.is_initialized():
                raise ValueError("Detector not initialized")
        
        else:
            self.detector = PSPokerDetect()
            self.detector.load_images()

        self.wm = None

        self.path_to_exec = path_to_exec

        self.ps_process = None

        # detect whether on windows or linux
        if os.name == "posix":
            self.linux = True
        else:
            self.linux = False

        



    def open_pokerstars(self):
        if self.linux:
             self.ps_process = subprocess.Popen(["gio", "launch", self.path_to_exec], stdout=None)
             

    def shutdown(self):
        if self.ps_process:
            self.ps_process.kill()
            self.ps_process = None




    def start(self, username: str, password: str):
        self.open_pokerstars()
        self.login(username, password)
        self.halt_until_room_select()
        time.sleep(4) # built-in delay/transition into the website



    def login(self, username: str, password: str):
        pyautogui.press("tab")
        pyautogui.write(username)
        pyautogui.press("tab")
        pyautogui.write(password)
        pyautogui.press("enter")

    def wait_until_in_room(self):
        while True:
            if not self.are_we_in_room():
                time.sleep(0.5)
                print("Halting until room select: Page is loading...")
            else:
                break


        
    def are_we_in_room(self):
        # first try, check for banner/header thingy on top of standard page
        # Loading...
        ids = get_all_windows_matching(".+No Limit Hold'em.+")
        return len(ids) > 0


  
        




    def __canvas_click(self, x: int, y: int, amt=1):
        body = self.driver.find_element(By.TAG_NAME, "body")
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        bW, bH = body.size["width"], body.size["height"]
        cX, cY = canvas.location["x"], canvas.location["y"]
        
        # ActionChains(self.driver).move_to_element_with_offset(body,-bW // 2 + x + cX, -bH // 2 + y + cY).click().perform()

        # do X times
        chain = ActionChains(self.driver)
        chain.move_to_element_with_offset(body, -bW // 2 + x + cX, -bH // 2 + y + cY)
        chain.click()
        for i in range(amt - 1):
            # chain.pause(0.1) # hey btw do you mind if i make the login thing use a config.json that isnt git tracked?
            chain.click()
        chain.perform()



    def __canvas_drag(self, x1: int, y1: int, x2: int, y2: int):
        body = self.driver.find_element(By.TAG_NAME, "body")
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        bW, bH = body.size["width"], body.size["height"]
        cX, cY = canvas.location["x"], canvas.location["y"]

        ActionChains(self.driver) \
            .move_to_element_with_offset(body, -bW // 2 + x1 + cX, -bH // 2 + y1 + cY) \
            .click_and_hold() \
            .move_by_offset(x2 - x1, y2 - y1) \
            .release().perform()

    def _ss(self):  # why not make this a class property/field that is updated once per tick? - amelia
        try:
            el = self.driver.find_element(By.TAG_NAME, "canvas")
            img = prepare_ss(el.screenshot_as_png)

            idx = int(time.time() * 100_000) / 100_000
            with open(f"./midrun/canvas-{idx}.png", "wb") as f:
                f.write(el.screenshot_as_png)

            with open(f"./midrun/ss-{idx}.png", "wb") as f:
                f.write(self.driver.get_screenshot_as_png())
            
            print(img.shape)
            return img
        except Exception as e:
            return None
    

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


        self.__canvas_click(*mid(press_plus), clicks)
        
        button = self.detector.bet_button(ss, threshold=0.8)
        if button is None:
            button = self.detector.raise_button(ss, threshold=0.8)
            if button is None:
                button = self.detector.allin_button(ss, threshold=0.8)
                if button is None:
                    return False

        self.__canvas_click(*mid(button))
        return True
    

    def reraise(self, amt: int, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.raise_button(ss, threshold=0.8)

        if button is None:
            return False

        self.__canvas_click(*mid(button))
        return True

    def check(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False

        if ss is None:
            ss = self._ss()

        button = self.detector.check_button(ss, threshold=0.8)

        if button is None:
            return False

        self.__canvas_click(*mid(button))
        return True

    def fold(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.fold_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.__canvas_click(*mid(button))
        return True

    def call(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False


        if ss is None:
            ss = self._ss()


        button = self.detector.call_button(ss, threshold=0.8)

        if button is None:
            return False
        
        self.__canvas_click(*mid(button))
        return True

    def sit(self) -> bool:
        pass



if __name__ == "__main__":
    detector = PSPokerDetect()
    detector.load_images()
    from ...all.windows import UnixWindowManager
    test = PSInteract(detector=detector, path_to_exec="/home/generel/.local/share/applications/wine/Programs/PokerStars.net/PokerStars.net.desktop")
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