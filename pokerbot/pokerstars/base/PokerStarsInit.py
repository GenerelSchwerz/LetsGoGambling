

import os
import subprocess

import pyautogui

import time


from ...abstract.pokerInit import PokerInitSetup
from ...all.windows import get_all_windows_matching,  UnixWindowManager

class PokerStarsInit(PokerInitSetup):

    def __init__(self, path_to_exec: str):
        self.path_to_exec = path_to_exec
        self.wm = None
        self.ps_process = None

        # detect whether on windows or linux
        if os.name == "posix":
            self.linux = True
        else:
            self.linux = False

    def open_pokerstars(self):
        if self.linux:
             self.ps_process = subprocess.Popen(["gio", "launch", self.path_to_exec], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             

    def shutdown(self):
        if self.ps_process:
            self.ps_process.kill()
            self.ps_process = None


    def start(self):
        self.open_pokerstars()


    def login(self, username: str, password: str):
        pyautogui.press("tab")
        pyautogui.write(username)
        pyautogui.press("tab")
        pyautogui.write(password)
        pyautogui.press("enter")


    def start_tables(self):
        return super().start_tables()

    def wait_until_in_room(self):
        while True:
            if not self.are_we_in_room():
                time.sleep(0.5)
                print("Halting until room select: Page is loading...")
            else:
                break
        
        if self.linux:
            ids = get_all_windows_matching(".+No Limit Hold'em.+")
            self.wm = UnixWindowManager(id=ids[0])
            # self.detector.load_wm(self.wm)


    def are_we_in_room(self):
        # first try, check for banner/header thingy on top of standard page
        # Loading...
        ids = get_all_windows_matching(".+No Limit Hold'em.+")
        return len(ids) > 0