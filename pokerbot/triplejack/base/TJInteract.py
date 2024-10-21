import random
from typing import Union
from ..driver_creator import create_tj_driver
from ...abstract.pokerInteract import PokerInteract
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement

from .TJPokerDetect import TJPokerDetect

from ..utils import prepare_ss

import time

from logging import Logger

log = Logger("PokerBot")

class TJPage:
    INVALID = -999
    UNKNOWN = -1
    LOGIN = 0
    LOADING = 1
    LOBBY = 2
    TABLE = 3

    lookup_table = {
        INVALID: "Invalid",
        UNKNOWN: "Unknown",
        LOGIN: "Login",
        LOADING: "Loading",
        LOBBY: "Lobby",
        TABLE: "Table"
    }

    @staticmethod
    def get_page(driver: webdriver.Chrome):
        if "triplejack.com" not in driver.current_url:
            return TJPage.INVALID
        
        # TODO: actual detection
        if "lobby" in driver.current_url:
            return TJPage.LOBBY
        
        # TODO: actual detection
        if "table" in driver.current_url:
            return TJPage.TABLE
        
        return TJPage.UNKNOWN
    
    @staticmethod
    def to_str(page: int):
        return TJPage.lookup_table.get(page, "Unknown")

    

def mid(tupl: tuple[int, int, int, int]) -> tuple[int, int]:
    return (tupl[0] + tupl[2]) // 2, (tupl[1] + tupl[3]) // 2

class TJInteract(PokerInteract):

    def __init__(self, driver, detector: Union[TJPokerDetect, None] = None):
        super().__init__()
        

        # performance in case multiple instances
        if detector:
            self.detector = detector
            if not self.detector.is_initialized():
                raise ValueError("Detector not initialized")
        
        else:
            self.detector = TJPokerDetect()
            self.detector.load_images()

        self.driver: webdriver.Chrome = driver
        self.page = TJPage.get_page(self.driver)

        self.call_location = None
        self.fold_location = None
        self.check_location = None
        self.bet_location = None
        self.raise_location = None
        self.allin_location = None
        self.plus_location = None

    def shutdown(self):
        return self.driver.quit()


    def try_close_popup(self):
        try:
            # identify by data-testid
            el = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='CloseIcon']")
            el.click()
            return True
        except Exception as e:
            return False
        

    


    def _canvas_click(self, x: int, y: int, amt=1):
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



    def _canvas_drag(self, x1: int, y1: int, x2: int, y2: int):
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

            # with open(f"./midrun/ss-{idx}.png", "wb") as f:
            #     f.write(self.driver.get_screenshot_as_png())
            
            # print(img.shape)
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

        if self.plus_location is None:
            press_plus = self.detector.plus_button(ss, threshold=0.8)
            self.plus_location = press_plus
        else:
            press_plus = self.plus_location
        if press_plus is None:
            print("Could not find plus button")
            return False


        self._canvas_click(*mid(press_plus), clicks)
        
        button = self.detector.bet_button(ss, threshold=0.8)
        if button is None:
            button = self.detector.raise_button(ss, threshold=0.8)
            if button is None:
                button = self.detector.allin_button(ss, threshold=0.8)
                if button is None:
                    return False

        self._canvas_click(*mid(button))
        return True
    

    def reraise(self, amt: int, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()
        button = self.detector.raise_button(ss, threshold=0.8)

        if button is None:
            return False

        self._canvas_click(*mid(button))
        return True

    def check(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False

        if ss is None:
            ss = self._ss()

        if self.check_location is None:
            button = self.detector.check_button(ss, threshold=0.8)
            self.check_location = button
        else:
            button = self.check_location

        if button is None:
            return False

        self._canvas_click(*mid(button))
        return True

    def fold(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False
        

        if ss is None:
            ss = self._ss()

        if self.fold_location is None:
            button = self.detector.fold_button(ss, threshold=0.8)
            self.fold_location = button
        else:
            button = self.fold_location

        if button is None:
            return False
        
        self._canvas_click(*mid(button))
        return True

    def call(self, ss=None) -> bool:
        # if self.page != TJPage.TABLE:
        #     return False


        if ss is None:
            ss = self._ss()


        button = self.detector.call_button(ss, threshold=0.8)

        if button is None:
            button = self.detector.allin_button(ss, threshold=0.8)
            if button is None:
                return False
        
        self._canvas_click(*mid(button))
        return True

    def sit(self) -> bool:
        if self.page != TJPage.LOBBY:
            return

        pass

