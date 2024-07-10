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

from .utils import prepare_ss
from .base import TJPokerDetect, report_info


from ..abstract.pokerEventHandler import PokerEvents, PokerStages
from .base.gameEvents import TJEventEmitter

import time
import random

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



log = Logger("PokerBot")


class NewPokerBot:

    currently_running = 0

    """
    This class is the main class for the poker bot. It will handle all the image processing and decision making.
    """

    def __init__(
        self, headless=False, debug=False, skip_cards=False, continuous=False, big_blind=200, **kwargs
    ):
        self.debug = debug
        self.skip_cards = skip_cards
        self.continuous = continuous
        self.big_blind = big_blind

        self.driver: Union[webdriver.Chrome, None] = None
        self.cur_ss = None

        # info about itself
        self.in_room = False
        self.is_sitting = False
        self.in_hand = False

        self.detector = TJPokerDetect()
        self.event_handler = TJEventEmitter(self.detector)

        self.headless = headless


    # TODO: Specify options for driver upon entry.
    def __enter__(self):
    
        NewPokerBot.currently_running += 1
        log.debug(f"Entering PokerBot {NewPokerBot.currently_running}")

        self.detector.load_images()
        options = webdriver.ChromeOptions()

    
        options.add_argument('--no-sandbox')
        options.add_argument('--use-angle=vulkan')
        options.add_argument('--enable-features=Vulkan')
        options.add_argument('--disable-vulkan-surface')
        options.add_argument('--enable-unsafe-webgpu')

        options.add_argument('--mute-audio')

        # set window size to standard 1080p
        # 
        
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument('--disable-infobars')

        if self.headless:
            options.add_argument('--headless=new')
            options.add_argument("--window-size=1920,1080")
            self.driver = webdriver.Chrome(options=options)
            # self.driver.maximize_window() # why is this setting to 4k lol, no need
        else:
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--start-maximized")
            self.driver = webdriver.Chrome(options=options)
        
        self.driver.set_window_size(1920, 1080)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Exiting PokerBot {NewPokerBot.currently_running}")
        NewPokerBot.currently_running -= 1
        self.driver.quit()

    def initialize(self, username: str, password: str):
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

        self.detector.load_images()

    # ====================
    # Utilities
    # ====================

    def are_we_loading(self):
        # first try, check for banner/header thingy on top of standard page
        # Loading...

        try:
            self.driver.find_element(By.CLASS_NAME, "css-1ogd7j7")
            return False
        except Exception as e:
            return True

    def canvas_screenshot(self, store=True, save_loc: Union[str, None] = None) -> bytes:
        var = self.find_canvas()
        if not var:
            return None
        var = var.screenshot_as_png
        if store:
            self.cur_ss = var
        if save_loc:
            with open(save_loc, "wb") as f:
                f.write(var)
        return var

    def click_on_canvas(self, x: int, y: int):
        body = self.driver.find_element(By.TAG_NAME, "body")
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        bW, bH = body.size["width"], body.size["height"]
        cX, cY = canvas.location["x"], canvas.location["y"]
        
        ActionChains(self.driver).move_to_element_with_offset(body,-bW // 2 + x + cX, -bH // 2 + y + cY).click().perform()

    def drag_on_canvas(self, x1: int, y1: int, x2: int, y2: int):
        body = self.driver.find_element(By.TAG_NAME, "body")
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        bW, bH = body.size["width"], body.size["height"]
        cX, cY = canvas.location["x"], canvas.location["y"]

        ActionChains(self.driver) \
            .move_to_element_with_offset(body, -bW // 2 + x1 + cX, -bH // 2 + y1 + cY) \
            .click_and_hold() \
            .move_by_offset(x2 - x1, y2 - y1) \
            .release().perform()


    # ====================
    # Game Logic
    # ====================

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

    def try_close_popup(self):
        # <svg class="MuiSvgIcon-root MuiSvgIcon-fontSizeMedium css-vubbuv" focusable="false" aria-hidden="true" viewBox="0 0 24 24" data-testid="CloseIcon"><path fill="currentColor" d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z"></path></svg>
        try:
            # identify by data-testid
            el = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='CloseIcon']")
            el.click()
            return True
        except Exception as e:
            return False

    def find_canvas(self):
        try:
            return self.driver.find_element(By.TAG_NAME, "canvas")
        except Exception as e:
            return None

    def halt_until_room_select(self):
        if self.in_room:
            return

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

        ss = self.canvas_screenshot(store=False)

        if ss is None:
            return

        ss_good = prepare_ss(ss)
        locations = self.detector.sit_buttons(ss_good)

        # pick top left location

        locations = sorted(locations, key=lambda x: (x[1], x[0]))

        # locations = [locations[0]]

        # for loc in locations:
        #     cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 200, 0), 2)

        if len(locations) == 0:
            log.error("Could not find sit button")
            return
        
   

        averages = [((pt[0] + pt[2]) // 2, ((pt[1] + pt[3]) // 2)) for pt in locations]
        selection = averages[random.choice(averages)]
        self.detector.set_seat_loc(selection)

        # cv2.rectangle(ss_good, (selection[0], selection[1]), (selection[0] + 10, selection[1] + 10), (0, 255, 0), 2)
        # cv2.imshow("Sit Buttons", ss_good)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        self.click_on_canvas(int(selection[0]), int(selection[1]))

        # multiple options appear here.

        # if we're getting a free buy in:
        # Free Rebuy - Get $

        # else, just sit
        try:
            el = WebDriverWait(self.driver, 2).until(
                EC.visibility_of_element_located((By.XPATH, "//*[text()='Sit']"))
            )

            el.click()
        except Exception as e:
            log.debug("Could not find sit button, probably already sat at table")
            return


def show_all_info(bot: NewPokerBot, save=True):
    ss = bot.canvas_screenshot(store=False)

    ss_good = prepare_ss(ss)

    sit_button = bot.detector.sit_buttons(ss_good)

    for loc in sit_button:
        cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    call_button = bot.detector.call_button(ss_good)

    if call_button is not None:
        cv2.rectangle(
            ss_good,
            (call_button[0], call_button[1]),
            (call_button[2], call_button[3]),
            (0, 255, 0),
            2,
        )

    raise_button = bot.detector.raise_button(ss_good)

    if raise_button is not None:
        cv2.rectangle(
            ss_good,
            (raise_button[0], raise_button[1]),
            (raise_button[2], raise_button[3]),
            (0, 255, 0),
            2,
        )

    fold_button = bot.detector.fold_button(ss_good)

    if fold_button is not None:
        cv2.rectangle(
            ss_good,
            (fold_button[0], fold_button[1]),
            (fold_button[2], fold_button[3]),
            (0, 255, 0),
            2,
        )

    check_button = bot.detector.check_button(ss_good)

    if check_button is not None:
        cv2.rectangle(
            ss_good,
            (check_button[0], check_button[1]),
            (check_button[2], check_button[3]),
            (0, 255, 0),
            2,
        )

    bet_button = bot.detector.bet_button(ss_good)

    if bet_button is not None:
        cv2.rectangle(
            ss_good,
            (bet_button[0], bet_button[1]),
            (bet_button[2], bet_button[3]),
            (0, 255, 0),
            2,
        )

    allin_button = bot.detector.allin_button(ss_good)

    if allin_button is not None:
        cv2.rectangle(
            ss_good,
            (allin_button[0], allin_button[1]),
            (allin_button[2], allin_button[3]),
            (0, 255, 0),
            2,
        )

    card_and_locs = bot.detector.community_cards_and_locs(ss_good)

    for card, loc in card_and_locs:
        cv2.putText(
            ss_good,
            Card.int_to_str(card),
            (loc[0], loc[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    card_and_locs1 = bot.detector.hole_cards_and_locs(ss_good)

    for card, loc in card_and_locs1:
        cv2.putText(
            ss_good,
            Card.int_to_str(card),
            (loc[0], loc[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    # cv2.imshow("all cards", ss_good)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if save:
        cv2.imwrite("all_cards.png", ss_good)

    return card_and_locs


def main():
    import time

    with NewPokerBot(headless=False) as bot:
        bot.initialize("Generel", "Rocky1928!")
        bot.halt_until_room_select()
        time.sleep(4) # built-in delay/transition into the website

        bot.sit_down() # sit down at table
        while bot.try_close_popup():
            time.sleep(0.25)


        def on_info(stage: PokerStages, hand: list[Card], community: list[Card]):
            # show_all_info(bot, save=True)
            print(PokerStages.to_str(stage), [Card.int_to_pretty_str(c) for c in hand], [Card.int_to_pretty_str(c) for c in community])
       
        def on_new_stage(from_stage: PokerStages, to_stage: PokerStages):
            print(PokerStages.to_str(from_stage), "=>", PokerStages.to_str(to_stage))

        def on_new_hand(*args):
            print("New Hand", *args)

        def on_our_turn(*args):
            print("Our Turn", *args)



        bot.event_handler.on(PokerEvents.NEW_HAND, on_new_hand)
        bot.event_handler.on(PokerEvents.NEW_STAGE, on_new_stage)
        bot.event_handler.on(PokerEvents.OUR_TURN, on_our_turn)

        bot.event_handler.on(PokerEvents.TICK, on_info)

        # main loop.

        time_split = 2
        last_time = time.time()
        while True:

            # input("take screenie ")
            img = bot.canvas_screenshot(store=False, save_loc="test.png") #
            img = prepare_ss(img)
            # report_info(bot.detector img)
            bot.event_handler.tick(img)

            now = time.time()
            dur = max(0, time_split - (now - last_time))
            print("Sleeping for ", dur, "been", now - last_time, "seconds taken to calculate since last tick.", dur + now - last_time, "seconds total")
            time.sleep(dur)
            last_time = time.time()

        time.sleep(600)



if __name__ == "__main__":
    main()
