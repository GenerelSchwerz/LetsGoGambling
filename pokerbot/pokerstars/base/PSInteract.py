
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

    def __init__(self, firefox=False, headless=False, detector: Union[TJPokerDetect, None] = None):
        super().__init__()
        

        # performance in case multiple instances
        if detector:
            self.detector = detector
            if not self.detector.is_initialized():
                raise ValueError("Detector not initialized")
        
        else:
            self.detector = TJPokerDetect()
            self.detector.load_images()

        self.driver = create_tj_driver(headless=headless, firefox=firefox)
        self.headless = headless

        self.page = TJPage.get_page(self.driver)

    def do_stupid_resizing(self):
        data = self.driver.get_window_size()
        width = data["width"]
        height = data["height"] * 9 // 10
        self.driver.execute_script(f"""
            var canvas = document.querySelector('.css-80txve');  // Adjust selector if necessary
            if (canvas) {{
                canvas.width = {width};  // Set desired width
                canvas.height = {height};  // Set desired height
                canvas.style.width = '{width}px';  // Set desired CSS width
                canvas.style.height = '{height}px';  // Set desired CSS height
            }} else {{
                console.error('Canvas not found');
            }}
            """)
            
        self.driver.set_window_size(data["width"] // 2, data["height"] // 2)
        self.driver.set_window_size(data["width"], data["height"])

        # now select canvas (active element)
        # self.driver.find_element(By.TAG_NAME, "body").click()

        


    def start(self, username: str, password: str):
        self.login(username, password)
        self.halt_until_room_select()
        time.sleep(4) # built-in delay/transition into the website

     
        # time.sleep(0.5)

        self.sit_down() # sit down at table
      
        while self.try_close_popup():
            time.sleep(0.25)

        self.do_stupid_resizing()


    def login(self, username: str, password: str):
        self.driver.get("https://triplejack.com")

        el = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "login-username-0"))
        )

        el.send_keys(username)

        el = self.driver.find_element(By.ID, "login-password-0")
        el.send_keys(password)

        # login button
        el = self.driver.find_element(By.CSS_SELECTOR, "#entry-panel > form > button")
        el.click()


    def find_room(self, opt: Union[str, int]) -> Union[WebElement, None]:
        # <h6 class="MuiTypography-root MuiTypography-h6 css-1uy08o2" id="lobby-room-50"><div>Time to Grind</div></h6>
        # identify element by its id, always starts with lobby-room

        try:
            elements = self.driver.find_elements(
                By.XPATH, "//*[starts-with(@id, 'lobby-room-')]"
            )

            if opt.isdigit():
                if 0 <= int(opt) < len(elements):
                    return elements[int(opt)]
                else:
                    return None

            elif isinstance(opt, str):
                for el in elements:
                    if opt.lower() in el.text.lower():
                        return el
                return None
            else:
                raise ValueError("Invalid type for opt")

        except Exception as e:
            return None
        
    def are_we_loading(self):
        # first try, check for banner/header thingy on top of standard page
        # Loading...

        try:
            self.driver.find_element(By.CLASS_NAME, "css-1ogd7j7")
            return False
        except Exception as e:
            return True

    def try_close_popup(self):
        try:
            # identify by data-testid
            el = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='CloseIcon']")
            el.click()
            return True
        except Exception as e:
            return False
        
    def get_all_room_names(self) -> list[str]:
        # <h6 class="MuiTypography-root MuiTypography-h6 css-1uy08o2" id="lobby-room-50"><div>Time to Grind</div></h6>
        # identify element by its id, always starts with lobby-room

        try:
            elements = self.driver.find_elements(
                By.XPATH, "//*[starts-with(@id, 'lobby-room-')]"
            )

            return [el.text for el in elements]

        except Exception as e:
            return []

        
    def halt_until_room_select(self):
            while True:
                if self.are_we_loading():
                    time.sleep(0.5)
                    log.debug("Halting until room select: Page is loading...")
                else:
                    break

            def print_rooms():
                rooms = self.get_all_room_names()

                print("Rooms available: ")
                for i, room in enumerate(rooms, 0):
                    print(f"{i}: {room}")

            print("Type 'skip' to skip room selection")
            print("Type 'rooms' to see available rooms")
            print_rooms()

            while True:
                room = input("Enter room name: ")
                log.debug(f"Attempting to find room {room}")

                if room == "skip": break

                if room == "rooms":
                    print_rooms()
                    continue

                room_el = self.find_room(room)
                if room_el:
                    while self.try_close_popup():
                        time.sleep(0.25)

                    log.debug(f"Found room {room}, joining...")
                    return room_el.click()

    def sit_down(self):
        while self.try_close_popup():
            time.sleep(0.25)

        ss = self._ss()
        locations = self.detector.sit_buttons(ss)

      

        # pick top left location

        locations = sorted(locations, key=lambda x: (x[1], x[0]))

   
        if len(locations) == 0:
            log.error("Could not find sit button")
            return
        
        averages = [((pt[0] + pt[2]) // 2, ((pt[1] + pt[3]) // 2)) for pt in locations]
        selection = random.choice(averages)

        self.__canvas_click(*selection)

        # multiple options appear here.

        # if we're getting a free buy in:
        # Free Rebuy - Get $
        # ^ TODO? - amelia

        # else, just sit
        try:
            el = WebDriverWait(self.driver, 2).until(
                EC.visibility_of_element_located((By.XPATH, "//*[text()='Sit']"))
            )

            el.click()
        except Exception as e:
            log.debug("Could not find sit button, probably already sat at table")
            return

    def shutdown(self):
        self.driver.quit()

    


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
        if self.page != TJPage.LOBBY:
            return

        pass

