import json
import time

from ..abstract.pokerEventHandler import PokerEvents

from ..all.abstractClient import AClient
from ..all.gameEvents import ImgPokerEventEmitter

from .base import CPokerImgDetect, CPokerInteract


import os
import subprocess

import pyautogui

import time

from ..all.abstractClient import AClient
from ..all.algoLogic import AlgoDecisions


from ..abstract.pokerInit import MultiTableSetup
from ..all.windows import get_all_windows_matching, UnixWindowManager

import logging
log = logging.getLogger(__name__)


def run_ticks(_bot: AClient, time_split=2):
    import cv2

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


def _print(*args):
    print(*args)


def start_table(window_id: int):
    wm = UnixWindowManager(window_id)
    detector = CPokerImgDetect()
    detector.load_wm(wm)
    detector.load_images()

    event_handler = ImgPokerEventEmitter(detector=detector)
    interactor = CPokerInteract(detector=detector, wm=wm)

    logic = AlgoDecisions()

    client = AClient(
        event_handler=event_handler,
        detector=detector,
        interactor=interactor,
        logic=logic,
        debug=True,
    )

    try:
        client.start()
        client.event_handler.on(PokerEvents.TICK, _print)
        run_ticks(client, 2)
    except KeyboardInterrupt:
        print("bot finished")
        client.stop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        client.stop()


class CPokerInit(MultiTableSetup):

    def __init__(self, path_to_exec: str):
        super().__init__()
        self.path_to_exec = path_to_exec
        self.parent_ps_process = None

        # detect whether on windows or linux
        if os.name == "posix":
            self.linux = True
        else:
            self.linux = False

    def open_pokerstars(self):
        if self.linux:
            self.parent_ps_process = subprocess.Popen(
                ["gio", "launch", self.path_to_exec],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def shutdown(self):
        if self.parent_ps_process:
            self.parent_ps_process.kill()
            self.parent_ps_process = None

    def start(self):
        self.open_pokerstars()

    def login(self, username: str, password: str):
        pyautogui.press("tab")
        pyautogui.press("tab")
        pyautogui.press("tab")
        pyautogui.write(username)
        pyautogui.press("tab")
        pyautogui.write(password)
        pyautogui.press("enter")

    def start_tables(self, amt: int = 1):
        # for i in range(amt):
        #     pyautogui.click(100, 100)
        #     time.sleep(1)

        windows = get_all_windows_matching(".+NL Hold'em.+")

        return [
            self.process_executor.submit(start_table, window_id)
            for window_id in windows
        ]


def main():
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s|%(name)s]: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # clear midrun

    os.makedirs("./midrun", exist_ok=True)

    for file in os.listdir("./midrun"):
        os.remove(f"./midrun/{file}")

    try:
        with open("pokerbot/config.json") as config_file:
            ps_cfg = json.load(config_file)["coinpoker"]
            acc = ps_cfg["account"]
            exec_path: str = ps_cfg["exec_path"]

    except FileNotFoundError:
        print("Config file not found")
        return

    try:
        full_client = CPokerInit(exec_path)
        # full_client.start()
        # time.sleep(15)
        # full_client.login(acc["username"], acc["password"])

        # time.sleep(20)

        clients = full_client.start_tables(1)
        full_client.wait_for_all(clients)
        full_client.shutdown()
    except KeyboardInterrupt:
        print("bot finished")
    
    finally:
        full_client.shutdown()


if __name__ == "__main__":
    main()
