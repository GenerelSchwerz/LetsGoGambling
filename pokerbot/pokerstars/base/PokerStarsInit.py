import os
import subprocess

import pyautogui

import time

from pokerbot.all.abstractClient import AClient
from pokerbot.all.algoLogic import AlgoDecisions
from pokerbot.pokerstars.base import PSInteract
from pokerbot.pokerstars.base.PSGameEvents import PSEventEmitter
from pokerbot.pokerstars.base.PSPokerDetect import PSPokerImgDetect


from ...abstract.pokerInit import MultiTableSetup
from ...all.windows import get_all_windows_matching, UnixWindowManager


class PokerStarsInit(MultiTableSetup):

    def __init__(self, path_to_exec: str):
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
        pyautogui.write(username)
        pyautogui.press("tab")
        pyautogui.write(password)
        pyautogui.press("enter")

    def start_tables(self, amt: int = 1) -> list[AClient]:
        for i in range(amt):
            pyautogui.click(100, 100)
            time.sleep(1)

        windows = get_all_windows_matching(".+No Limit Hold'em.+")

        clients = []
        for window in windows:
            wm = UnixWindowManager(window)
            detector = PSPokerImgDetect()
            detector.load_wm(wm)
            detector.load_images()
            event_handler = PSEventEmitter(detector=detector)

            interactor = PSInteract(detector=detector, wm=wm, path_to_exec=self.path_to_exec)

            logic = AlgoDecisions()

            clients.append(AClient(
                event_handler=event_handler,
                detector=detector,
                interactor=interactor,
                logic=logic,
                debug=True,
            ))

        return clients
