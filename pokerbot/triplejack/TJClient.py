from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import random
import time
from typing import Union
from pokerbot.abstract.pokerDecisions import PokerDecisionChoice

from pokerbot.triplejack.driver_creator import create_tj_driver

from ..all.abstractClient import AClient
from .base import TJPokerDetect, TJInteract, TJEventEmitter


import json
import time

from ..abstract.pokerEventHandler import PokerEvents

from ..all.abstractClient import AClient
from .base import TJPokerDetect, TJInteract, TJEventEmitter


import os
import subprocess

import pyautogui

import time

from pokerbot.all.abstractClient import AClient
from pokerbot.all.algoLogic import AlgoDecisions
from pokerbot.pokerstars.base.PSGameEvents import PSEventEmitter
from pokerbot.pokerstars.base.PSPokerDetect import PSPokerImgDetect


from ..abstract.pokerInit import MultiTableSetup


from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement


from logging import Logger

log = Logger("PokerBot")

class TJInit:

    def __init__(self, headless=False, firefox=False):
        self.headless = headless
        self.firefox = firefox
        self.logic = AlgoDecisions()

    def shutdown(self):
        self.interactor.shutdown()

    def do_stupid_resizing(self):
        data = self.driver.get_window_size()
        width = data["width"]
        height = data["height"] * 9 // 10
        self.driver.execute_script(
            f"""
            var canvas = document.querySelector('.css-80txve');  // Adjust selector if necessary
            if (canvas) {{
                canvas.width = {width};  // Set desired width
                canvas.height = {height};  // Set desired height
                canvas.style.width = '{width}px';  // Set desired CSS width
                canvas.style.height = '{height}px';  // Set desired CSS height
            }} else {{
                console.error('Canvas not found');
            }}
            """
        )

        self.driver.set_window_size(data["width"] // 2, data["height"] // 2)
        self.driver.set_window_size(data["width"], data["height"])

        # now select canvas (active element)
        # self.driver.find_element(By.TAG_NAME, "body").click()

    def start(self, username: str, password: str):

        self.detector = TJPokerDetect(username)
        self.detector.load_images()

        self.driver = create_tj_driver(headless=self.headless, firefox=self.firefox)

        self.interactor = TJInteract(detector=self.detector, driver=self.driver)

        self.login(username, password)

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

            if room == "skip":
                break

            if room == "rooms":
                print_rooms()
                continue

            room_el = self.find_room(room)
            if room_el:
                while self.interactor.try_close_popup():
                    time.sleep(0.25)

                log.debug(f"Found room {room}, joining...")
                return room_el.click()

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

    def sit_down(self):
        while self.interactor.try_close_popup():
            time.sleep(0.25)

        ss = self.interactor._ss()
        locations = self.detector.sit_buttons(ss)

        # pick top left location

        locations = sorted(locations, key=lambda x: (x[1], x[0]))

        if len(locations) == 0:
            log.error("Could not find sit button")
            return

        averages = [((pt[0] + pt[2]) // 2, ((pt[1] + pt[3]) // 2)) for pt in locations]
        selection = random.choice(averages)

        self.interactor._canvas_click(*selection)

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

    def start_table(self):

        self.halt_until_room_select()
        time.sleep(4)  # built-in delay/transition into the website

        # time.sleep(0.5)

        self.sit_down()  # sit down at table

        while self.interactor.try_close_popup():
            time.sleep(0.25)

        self.do_stupid_resizing()

        print("here!")

        event_handler = TJEventEmitter(detector=self.detector)

        client = AClient(
            event_handler=event_handler,
            detector=self.detector,
            interactor=self.interactor,
            logic=self.logic,
            debug=True,
        )

        client.start()

        return client


def run_ticks(_bot: AClient, time_split=2):
    import cv2

    print("hey1")

    last_time = time.time()
    while True:
        ss = _bot.interactor._ss()

        # cv2.imwrite(f"./midrun/{int(time.time() * 100_000) / 100_000}.png", ss)

        _bot.event_handler.tick(ss)
        now = time.time()
        dur = max(0, time_split - (now - last_time))
        print(
            "Sleeping for",
            round(dur * 1000) / 1000,
            "seconds. Been",
            round((now - last_time) * 1000) / 1000,
            "seconds taken to calculate since last tick.",
            round((dur + now - last_time) * 1000) / 1000,
            "seconds total",
        )
        time.sleep(dur)
        last_time = time.time()


def launch_user(
    username: str, password: str, time_split=2, headless=False, firefox=False
):
    full_client = TJInit(headless=headless, firefox=firefox)
    full_client.start(username, password)

    try:
        print("hey")
        client = full_client.start_table()
        run_ticks(client, time_split)

    except KeyboardInterrupt:
        print("bot finished")
        client.stop()

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)


def main():
    import os

    # clear midrun

    os.makedirs("./midrun", exist_ok=True)

    # for file in os.listdir("./midrun"):
    #     os.remove(f"./midrun/{file}")

    try:
        with open("pokerbot/config.json") as config_file:
            tj_cfg = json.load(config_file)["triplejack"]
            accAmt = tj_cfg.get("accountAmt", 1)
            accs = tj_cfg["accounts"]

    except FileNotFoundError:
        print("Config file not found")
        return

    if accAmt > 1:
        with ProcessPoolExecutor() as executor:
            for idx in range(accAmt):
                info = accs[idx]
                username = info["username"]
                password = info["password"]
                time_split = tj_cfg["secondsPerTick"]
                headless = tj_cfg["headless"]
                firefox = tj_cfg["firefox"]
                executor.submit(
                    launch_user, username, password, time_split, headless, firefox
                )
    else:
        username = accs[0]["username"]
        password = accs[0]["password"]
        time_split = tj_cfg["secondsPerTick"]
        headless = tj_cfg["headless"]
        firefox = tj_cfg["firefox"]
        launch_user(username, password, time_split, headless, firefox)


    # print all errors

    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
