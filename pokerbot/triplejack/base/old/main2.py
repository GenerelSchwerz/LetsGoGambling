import math
from typing import Self, Union

import pyautogui
import pytesseract
from PIL import ImageGrab, Image, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Resampling
from scipy.stats import gaussian_kde

from PIL import Image, ImageEnhance
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement


from logging import Logger

from newImgDetect import PokerImgDetect
import time
import random

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_holecard_locations(screenshot, expected_amt=2) -> list[tuple[int, int]]:
    pass


def detect_communitycard_locations(screenshot, expected_amt=5) -> list[tuple[int, int]]:
    pass


log = Logger("PokerBot")


class NewPokerBot:

    currently_running = 0

    """
    This class is the main class for the poker bot. It will handle all the image processing and decision making.
    """

    def __init__(
        self, debug=False, skip_cards=False, continuous=False, big_blind=200, **kwargs
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

        self.detector = PokerImgDetect("tests/")

    # TODO: Specify options for driver upon entry.
    def __enter__(self, tes_path: Union[str, None] = None, **kwargs) -> Self:
        if tes_path:
            if NewPokerBot.currently_running == 0:
                pytesseract.pytesseract.tesseract_cmd = tes_path
                NewPokerBot.currently_running = 1
            elif tes_path != pytesseract.pytesseract.tesseract_cmd:
                log.error(
                    "Tesseract path already set, cannot change it"
                )  # redundant but whatever
                raise ValueError("Tesseract path already set, cannot change it")
            pytesseract.pytesseract.tesseract_cmd = tes_path

        NewPokerBot.currently_running += 1
        log.debug(f"Entering PokerBot {NewPokerBot.currently_running}")

        self.detector.load_images()
        args = webdriver.ChromeOptions()
        # args.add_argument("--use-gl=shim")
        args.add_argument("--disable-gpu")
        # 

        # set window size to standard 1080p
        args.add_argument("--window-size=1920,1080")
        args.add_argument("--start-maximized")
        # args.add_argument("--headless")

        self.driver = webdriver.Chrome(options=args)

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

        self.driver.save_screenshot("test.png")


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
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")

        ActionChains(self.driver).move_to_element_with_offset(
            canvas, -canvas.size["width"] // 2, -canvas.size["height"] // 2
        ).move_by_offset(x, y).click().perform()

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
                else: return None

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

    
        rooms = self.get_all_room_names()

        print("Rooms available: ")
        for i, room in enumerate(rooms):
            print(f"{i}: {room}")

        while True:
            room = input("Enter room name: ")
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
        ss_good = self.detector.prepare_ss(ss)
        locations = self.detector.detect_sit_button(ss_good)

        # pick top left location

        locations = sorted(locations, key=lambda x: (x[1], x[0]))

        # locations = [locations[0]]

        # for loc in locations:
        #     cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

        if len(locations) == 0:
            log.error("Could not find sit button")
            return

        averages = [((pt[0] + pt[2]) // 2, ((pt[1] + pt[3]) // 2)) for pt in locations]
        selection = averages[random.randint(0, len(averages) - 1)]

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

    def get_all_cards(self):
        ss = self.canvas_screenshot(store=False, save_loc="test.png")
        ss_good = self.detector.prepare_ss(ss)

        community_locs = self.detector.community_suits(ss_good, threshold=0.77)

        print(community_locs)

        for key, suit_pos in community_locs.items():

            for loc in suit_pos:
                text = self.detector.card_number(ss_good, loc)
                text = f"{key} {text}"
                cv2.putText(ss_good, text, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)


        hole_locs = self.detector.hole_suits(ss_good, threshold=0.85)

        print(hole_locs)

        for key, suit_pos in hole_locs.items():
            for loc in suit_pos:
                text = self.detector.card_number(ss_good, loc)
                text = f"{key} {text}"
                cv2.putText(ss_good, text, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(ss_good, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)


        cv2.imshow("all cards", ss_good)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        return community_locs


def main():
    import time

    with NewPokerBot() as bot:
        bot.initialize("ForTheChips", "WooHoo123!")
        bot.halt_until_room_select()
        time.sleep(4)
        bot.sit_down()
        while True:
            bot.get_all_cards()
            

        time.sleep(600)


class PokerBot:
    def __init__(self):
        self.debug = False  # add debug windows and console messages
        self.skip_cards = False  # skip card detection
        self.continuous = False  # determines whether it'll loop or just run once
        self.big_blind = 200  # it will automatically determine this by constantly checking for the minimum bet
        # but i specify it here just so it doesn't potentially bug out the first hand or two

        self.screenshot = None

        # please don't make fun of me i just wanted this done
        self.holecard1_value_location = (45, 890, 52 + 70, 890 + 78)
        self.holecard1_suit_location = (42, 956, 42 + 74, 960 + 80)
        self.holecard2_value_location = (160, 890, 170 + 70, 890 + 78)
        self.holecard2_suit_location = (153, 956, 160 + 70, 960 + 80)
        self.board_card_locations = [
            [(455, 472, 463 + 57, 480 + 55), (458, 475 + 51, 468 + 65, 480 + 72 + 50)],
            [(550, 468, 561 + 55, 468 + 55), (550, 468 + 51, 561 + 65, 468 + 72 + 50)],
            [(645, 466, 654 + 55, 466 + 55), (640, 466 + 51, 644 + 65, 466 + 72 + 50)],
            [(735, 467, 748 + 55, 467 + 55), (728, 467 + 51, 748 + 65, 467 + 72 + 50)],
            [(825, 472, 836 + 55, 477 + 55), (817, 477 + 51, 836 + 65, 477 + 72 + 50)],
        ]
        # please just ignore all of this rocco please
        adjustment_factor = -37
        for i in range(len(self.board_card_locations)):
            for j in range(len(self.board_card_locations[i])):
                self.board_card_locations[i][j] = (
                    self.board_card_locations[i][j][0],
                    self.board_card_locations[i][j][1] + adjustment_factor,
                    self.board_card_locations[i][j][2],
                    self.board_card_locations[i][j][3] + adjustment_factor,
                )
        # this is my greatest shame
        adjustment_factor = -20
        self.holecard1_value_location = (
            self.holecard1_value_location[0],
            self.holecard1_value_location[1] + adjustment_factor,
            self.holecard1_value_location[2],
            self.holecard1_value_location[3] + adjustment_factor,
        )
        self.holecard1_suit_location = (
            self.holecard1_suit_location[0],
            self.holecard1_suit_location[1] + adjustment_factor,
            self.holecard1_suit_location[2],
            self.holecard1_suit_location[3] + adjustment_factor,
        )
        self.holecard2_value_location = (
            self.holecard2_value_location[0],
            self.holecard2_value_location[1] + adjustment_factor,
            self.holecard2_value_location[2],
            self.holecard2_value_location[3] + adjustment_factor,
        )
        self.holecard2_suit_location = (
            self.holecard2_suit_location[0],
            self.holecard2_suit_location[1] + adjustment_factor,
            self.holecard2_suit_location[2],
            self.holecard2_suit_location[3] + adjustment_factor,
        )

        # this is all the bot has when it comes to "memory"
        # street is the last street it remembers, not currently used
        # threshold is the minimum threshold it has used during the hand (reset every preflop)
        self.flags = {"street": 0, "threshold": 9}

        # initializing suit templates
        suits = ["s", "c", "h", "d"]
        hole_imgs = [
            "holespade.png",
            "holeclub.png",
            "holeheart.png",
            "holediamond.png",
        ]
        board_imgs = [
            "boardspade.png",
            "boardclub.png",
            "boardheart.png",
            "boarddiamond.png",
        ]
        self.hole_suits = {}
        self.board_suits = {}

        for i, suit in enumerate(suits):
            img_name = hole_imgs[i]
            suit_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            _, suit_img = cv2.threshold(
                suit_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.hole_suits[suit] = suit_img
            # suit_img = self.erase_edges(suit_img)
            # Display the binary image
            # plt.imshow(suit_img, cmap='gray')
            # plt.show()
            img_name = board_imgs[i]
            suit_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            _, suit_img = cv2.threshold(
                suit_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.board_suits[suit] = suit_img

    def take_screenshot(self):
        self.screenshot = ImageGrab.grab(xdisplay=":0")

    def click_element(self, location, clicks=1):
        # click at average X and Y of location tuple
        pyautogui.click(
            (location[0] + location[2]) // 2,
            (location[1] + location[3]) // 2,
            clicks=clicks,
            interval=0.01,
        )
        # move mouse to center of screen
        pyautogui.moveTo(pyautogui.size()[0] // 2, pyautogui.size()[1] // 2)

    def drag_slider(self, image_path, x_offset, y_offset=0):
        slider = self.find_element(image_path)
        if slider:
            pyautogui.moveTo((slider[0] + slider[2]) // 2, (slider[1] + slider[3]) // 2)
            pyautogui.dragTo(
                (slider[0] + slider[2]) // 2 + x_offset,
                (slider[1] + slider[3]) // 2 + y_offset,
                duration=0.5,
            )
        else:
            print(f"Could not find {image_path}")

    def erase_edges(self, binary_image):
        old_image = binary_image.copy()
        target_value = 0
        replacement_value = 255

        # not recursive bc stack overflow
        def flood_fill(binary_image, i, j):
            stack = [(i, j)]
            while stack:
                i, j = stack.pop()
                if (
                    i < 0
                    or i >= binary_image.shape[0]
                    or j < 0
                    or j >= binary_image.shape[1]
                ):
                    continue
                if binary_image[i, j] != target_value:
                    continue
                binary_image[i, j] = replacement_value
                stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

        # Flood fill from each side
        for i in range(binary_image.shape[0]):
            flood_fill(binary_image, i, 0)
            flood_fill(binary_image, i, binary_image.shape[1] - 1)
        for j in range(binary_image.shape[1]):
            flood_fill(binary_image, 0, j)
            flood_fill(binary_image, binary_image.shape[0] - 1, j)

        # check if whole image is white
        if np.all(binary_image == 255):
            print("Erased edges flood fill failed, whole image is white")
            return old_image
        else:
            return binary_image

    def ocr_text_from_image(
        self,
        location,
        rotation_angle=0,
        psm=7,
        blur_size=3,
        invert=False,
        colored_text=False,
        erode=False,
        brightness=3.0,
        contrast=0.0,
        hoz_stretch=0.0,
    ):
        # if self.debug:
        # show the location subsection of the screenshot
        # cv2.imshow('location', np.array(self.screenshot.crop(location)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image = np.array(self.screenshot.crop(location))

        if rotation_angle != 0:
            image = Image.fromarray(image)
            image = image.rotate(
                rotation_angle, resample=Resampling.BICUBIC, fillcolor=(255, 255, 255)
            )
            image = np.array(image)

        if not invert and not colored_text:
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

        if contrast > 0:
            img = Image.fromarray(np.uint8(image))
            contrasted_img = ImageEnhance.Contrast(img).enhance(contrast)
            image = np.array(contrasted_img)

        if (
            hoz_stretch > 0
        ):  # don't ask why this increases detection rate for 10's and 6's
            image = Image.fromarray(image)
            image = image.resize(
                (round(hoz_stretch * image.width), image.height),
                resample=Resampling.BICUBIC,
            )
            image = np.array(image)

        if self.debug:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # save image
            cv2.imwrite("image.png", image)

        # check if at least 80% of the pixels are the same color as the center pixel
        # otherwise it'll sometimes return a 0 and fuck things up
        center_pixel = image[image.shape[0] // 2, image.shape[1] // 2]
        distances = np.linalg.norm(image.astype(float) - center_pixel, axis=2)
        if np.sum(distances <= 20) / (image.shape[0] * image.shape[1]) > 0.93:
            if self.debug:
                print(
                    "Image is mostly the same color as the center pixel, returning empty string"
                )
            return ""

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # # show the gray
        # if self.debug:
        #     cv2.imshow('gray', gray)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # if blur_size != 1:
        #     gray = cv2.bitwise_not(gray)
        #     gray = cv2.medianBlur(gray, blur_size)
        #     gray = cv2.bitwise_not(gray)

        # downscale by 2
        # gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        # scale height to 33 pixels using bicubic resampling
        gray = cv2.resize(
            gray,
            (0, 0),
            fx=35 / gray.shape[0],
            fy=35 / gray.shape[0],
            interpolation=cv2.INTER_CUBIC,
        )

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if invert:
            binary = cv2.bitwise_not(binary)

        if self.debug:
            cv2.imshow("binary", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        binary = self.erase_edges(binary)

        if blur_size != 1:
            binary = cv2.bitwise_not(binary)
            binary = cv2.medianBlur(binary, blur_size)
            binary = cv2.bitwise_not(binary)

        # i know this effect better as feathering but everyone calls it erosion
        if erode:
            binary = cv2.bitwise_not(binary)
            binary = cv2.resize(binary, (0, 0), fx=2, fy=2)
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.resize(binary, (0, 0), fx=0.5, fy=0.5)
            binary = cv2.bitwise_not(binary)

        # Prepare a padding color
        color = [255, 255, 255]
        # Add padding around the binary image
        binary = cv2.copyMakeBorder(
            binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color
        )

        if self.debug:
            cv2.imshow("binary", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # save the image
            cv2.imwrite("binary.png", binary)

        allowed_chars = "0123456789,kMAKQJO"
        custom_config = (
            f"--oem 3 --psm {str(psm)} -c tessedit_char_whitelist=" + allowed_chars
        )
        result = pytesseract.image_to_string(
            binary, lang="eng", config=custom_config
        ).strip()
        if self.debug:
            print(f"OCR result: {result}")
        # can you tell what number has given me hours of trouble with tesseract and TJ font
        return (
            "10"
            if any(
                result == char
                for char in ["0", "O", "1", "I", "70", "1O", "IO", "I0", "7O"]
            )
            else result
        )

    def find_and_click(self, image_path, clicks=1):
        element = self.find_element(image_path)
        if element:
            self.click_element(element, clicks)
            return True
        else:
            if self.debug:
                print(f"Could not find {image_path}")
            return False

    def find_element(self, image_path, search_area=None, confidence_threshold=0.9):
        template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if search_area:
            template = template[
                search_area[1] : search_area[3], search_area[0] : search_area[2]
            ]
        screenshot = cv2.cvtColor(np.array(self.screenshot), cv2.COLOR_RGB2GRAY)
        confidence, location = self.min_max_loc(template, screenshot)
        if confidence > confidence_threshold:
            return location
        else:
            return None

    # returns list of location each in the form of (left, top, right, bottom)
    def find_elements(
        self, image_path, search_area=None, confidence_threshold=0.9, block_wide=False
    ):
        # search for all occurrences of the image and return a list of them
        template = cv2.imread(image_path, cv2.IMREAD_COLOR)

        screenshot = (
            self.screenshot.crop(search_area) if search_area else self.screenshot
        )

        res = cv2.matchTemplate(np.array(screenshot), template, cv2.TM_CCOEFF_NORMED)

        # if self.debug:
        #     cv2.imshow('res', res)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        locations = []
        while True:
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < confidence_threshold:
                break
            if not search_area:
                locations.append(
                    (
                        max_val,
                        (
                            max_loc[0],
                            max_loc[1],
                            max_loc[0] + template.shape[1],
                            max_loc[1] + template.shape[0],
                        ),
                    )
                )
            else:
                locations.append(
                    (
                        max_val,
                        (
                            max_loc[0] + search_area[0],
                            max_loc[1] + search_area[1],
                            max_loc[0] + search_area[0] + template.shape[1],
                            max_loc[1] + search_area[1] + template.shape[0],
                        ),
                    )
                )

            # show the location subsection of the screenshot
            # if self.debug:
            #     print(max_val)
            #     print(image_path)
            #     cv2.imshow('location', np.array(self.screenshot.crop(locations[-1])))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            res[
                max_loc[1] : max_loc[1] + template.shape[0],
                max_loc[0] : max_loc[0] + template.shape[1],
            ] = 0
            # zero out surrounding area too
            res[
                max_loc[1] - 10 : max_loc[1] + template.shape[0] + 10,
                max_loc[0] - 10 : max_loc[0] + template.shape[1] + 10,
            ] = 0
            # zero out a thin stretch of wide area about 200 pixels wide but as tall as the template
            if block_wide:
                res[
                    max_loc[1] - 10 : max_loc[1] + template.shape[0] + 10,
                    max_loc[0] - 200 : max_loc[0] + template.shape[1] + 200,
                ] = 0

            # show new res
            # if self.debug:
            #     cv2.imshow('res', res)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
        return locations

    def min_max_loc(self, template, image):
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # return confidence, (left, top, right, bottom)
        return max_val, (
            max_loc[0],
            max_loc[1],
            max_loc[0] + template.shape[1],
            max_loc[1] + template.shape[0],
        )

    def get_card_value_and_suit(
        self, value_location, suit_location, hole_card, board_card_number=0
    ):
        angles = {1: -5, 2: -3, 3: 0, 4: 3, 5: 5}

        card_value = self.ocr_text_from_image(
            value_location,
            rotation_angle=4 if hole_card else angles[board_card_number],
            psm=10,
            blur_size=1,
            brightness=3,
            contrast=2,
            hoz_stretch=1.1 if not hole_card else 0.0,
            erode=True,
        )
        if not hole_card and card_value not in [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "J",
            "Q",
            "K",
            "A",
        ]:
            print(f"Board card {board_card_number} not visible ({card_value})")
            return 0, "0"
        else:
            card_value = self.map_card_value(card_value)
            print(f"Found card with value: {card_value}")

        if self.debug:
            cv2.imshow("suit_location", np.array(self.screenshot.crop(suit_location)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cropped_screenshot = self.screenshot.crop(suit_location)

        if board_card_number != 0:
            # rotate according to angles[board_card_number] with bicubic sampling
            cropped_screenshot = cropped_screenshot.rotate(
                angles[board_card_number],
                resample=Resampling.BICUBIC,
                fillcolor=(255, 255, 255),
            )

        processed_suit = np.array(cropped_screenshot)
        processed_suit = cv2.cvtColor(processed_suit, cv2.COLOR_RGB2GRAY)
        _, processed_suit = cv2.threshold(
            processed_suit, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # processed_suit = self.erase_edges(processed_suit)
        processed_suit = cv2.copyMakeBorder(
            processed_suit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        # Display the binary image
        # if self.debug:
        #     plt.imshow(processed_suit, cmap='gray')
        #     plt.show()

        suits = ["s", "c", "h", "d"]

        confidences = {}
        for i, suit in enumerate(suits):
            suit_img = self.hole_suits[suit] if hole_card else self.board_suits[suit]

            confidence, _ = self.min_max_loc(suit_img, processed_suit)

            confidences[suit] = confidence

        card_suit, _ = max(confidences.items(), key=lambda x: x[1])

        if confidences[card_suit] < 0.85:
            card_suit = "unknown"
            print(f"Could not find suit with enough confidence")
            for suit, confidence in confidences.items():
                print(f"{suit}: {confidence}")
        print(
            f"Found card with suit: {card_suit} (Confidence: {confidences[card_suit]})"
        )

        return card_value, card_suit

    def map_card_value(self, card_value):
        card_value_dict = {"J": 11, "Q": 12, "K": 13, "A": 14}
        # throw exception if card_value isn't a number or a valid card value
        if not card_value.isdigit() and card_value not in card_value_dict:
            raise ValueError(f'Invalid card value: "{card_value}"')

        if card_value in card_value_dict:
            return card_value_dict[card_value]
        else:
            return int(card_value)

    def str_to_int(self, number):
        # remove comma
        has_period = "." in number or "," in number
        new_number = number.replace(",", "")
        new_number = new_number.replace(".", "")
        # replace 'k' with '000'
        new_number = new_number.replace("k", "00" if has_period else "000")
        new_number = new_number.replace("K", "00" if has_period else "000")
        try:
            return int(new_number)
        except ValueError:
            print(f"Could not convert number to int: {number} ({new_number})")
            return 0

    def main(self):
        iterations = 0
        while True:
            self.take_screenshot()

            current_bet, my_turn = self.get_bet_and_my_turn()

            if my_turn:
                print("\n\n")
                pyautogui.sleep(0.5)  # let some animations finish up
                self.take_screenshot()

                if not self.skip_cards:
                    board_cards, holecard1, holecard2 = self.get_cards()
                else:
                    board_cards, holecard1, holecard2 = [], (0, "0"), (0, "0")

                stack_size = self.get_stack_size()

                pot_value, middle_pot_value = self.get_pot(stack_size)

                num_opponents = self.get_num_opponents()

                print(f"Current bet: {current_bet}")
                min_bet = self.get_min_bet(current_bet, stack_size)

                if (
                    not (board_cards == [] and current_bet >= 10 * self.big_blind)
                    and not current_bet >= stack_size
                    and not (
                        not current_bet <= self.big_blind * 3
                        and current_bet >= middle_pot_value
                    )
                ):
                    approximate_pot = math.floor(
                        ((num_opponents + 1) * current_bet)
                        / (1.5 if current_bet != self.big_blind else 1)
                    )
                    if pot_value < approximate_pot:
                        print(
                            f"Pot value less than {approximate_pot}, setting it to that"
                        )
                        pot_value = approximate_pot

                if not board_cards:
                    print("PREFLOP")
                    game_stage = 0
                elif len(board_cards) == 3:
                    print("FLOP")
                    game_stage = 1
                elif len(board_cards) == 4:
                    print("TURN")
                    game_stage = 2
                else:
                    print("RIVER")
                    game_stage = 3

                if game_stage == 0:
                    self.flags["threshold"] = 9

                self.flags["street"] = game_stage

                print(f"Flags: {self.flags}")

                decision = make_decision(
                    [holecard1, holecard2],
                    board_cards,
                    stack_size,
                    pot_value,
                    current_bet,
                    min_bet,
                    num_opponents,
                    self.big_blind,
                    middle_pot_value,
                    self.flags,
                )

                print(f"Decision: {decision}")

                if decision:
                    if self.continuous:
                        self.wait_random_time(
                            decision, stack_size, pot_value, game_stage
                        )
                    if decision == "fold":
                        if not self.find_and_click("foldbutton.png"):
                            print("ERROR: Could not find fold button")
                    elif decision == "call":
                        if not self.find_and_click("callbutton.png"):
                            if not self.find_and_click("checkbutton.png"):
                                if not self.find_and_click("allinbutton.png"):
                                    print(
                                        "ERROR: Could not find call, check, or allin button"
                                    )
                    elif decision.startswith("bet"):
                        bet_amount = int(decision.split()[1])
                        if bet_amount >= stack_size:
                            self.drag_slider("slider.png", 250)
                        else:
                            small_blind = self.big_blind // 2
                            clicks = int(
                                (
                                    small_blind
                                    * round((bet_amount - min_bet) / small_blind)
                                )
                                / (self.big_blind // 2)
                            )
                            clean_clicks = clicks
                            # since we bet pot a lot, can get kinda sus if we bet exactly pot when pot is like 247BB
                            if clicks > 20:
                                clicks = int(clicks * (0.9 + (0.2 * random.random())))
                            print(
                                f"Clicking {clicks} times (varied from {clean_clicks})"
                            )
                            if clicks > 0:
                                if not self.find_and_click("plusbutton.png", clicks):
                                    print("ERROR: Could not find plus button")
                        if not self.find_and_click("betbutton.png"):
                            if not self.find_and_click("allinbutton.png"):
                                if not self.find_and_click("raisetobutton.png"):
                                    print(
                                        "ERROR: Could not find bet, allin, or raiseto button"
                                    )
                    else:
                        print(f"Invalid decision: {decision}")
            else:
                if iterations % 10 == 0:
                    print("Not my turn...")
            iterations += 1
            if not self.continuous:
                break
            else:
                pyautogui.sleep(0.5)

    def get_min_bet(self, current_bet, stack_size):
        if current_bet >= stack_size:
            print("Allin or fold...")
            min_bet = 0
        else:
            bet = self.find_element("betbutton.png")
            if bet:
                min_bet = self.str_to_int(
                    self.ocr_text_from_image(
                        (bet[0], bet[3] - 5, bet[2], bet[3] + (bet[3] - bet[1])),
                        blur_size=1,
                        colored_text=True,
                        contrast=2,
                    )
                )
            else:
                raiseto = self.find_element("raisetobutton.png")
                if raiseto:
                    min_bet = self.str_to_int(
                        self.ocr_text_from_image(
                            (
                                raiseto[0],
                                raiseto[3] - 5,
                                raiseto[2],
                                raiseto[3] + (raiseto[3] - raiseto[1]),
                            ),
                            blur_size=1,
                            colored_text=True,
                            contrast=2,
                        )
                    )
                else:
                    allin = self.find_element("allinbutton.png")
                    if allin:
                        min_bet = self.str_to_int(
                            self.ocr_text_from_image(
                                (
                                    allin[0],
                                    allin[3] - 5,
                                    allin[2],
                                    allin[3] + (allin[3] - allin[1]),
                                ),
                                blur_size=1,
                                colored_text=True,
                                contrast=2,
                            )
                        )
                    else:
                        print("ERROR: Could not find min bet")
                        min_bet = 0

            print(f"Min bet: {min_bet}")
            if self.big_blind > min_bet:
                self.big_blind = min_bet
        if current_bet == min_bet:
            raise ValueError("Current bet is equal to min bet")
        return min_bet

    def get_bet_and_my_turn(self):
        my_turn = False
        check = self.find_element("checkbutton.png")
        if check:
            current_bet = 0
            my_turn = True
        else:
            call = self.find_element("callbutton.png")
            if call:
                current_bet = self.str_to_int(
                    self.ocr_text_from_image(
                        (
                            call[0] - 5,
                            call[3] - 5,
                            call[2] + 5,
                            call[3] + (call[3] - call[1]),
                        ),
                        blur_size=1,
                        colored_text=True,
                        contrast=3,
                    )
                )
                my_turn = True
            else:
                allin = self.find_element("allinbutton.png")
                if allin:
                    current_bet = self.str_to_int(
                        self.ocr_text_from_image(
                            (
                                allin[0] - 5,
                                allin[3] - 5,
                                allin[2] + 5,
                                allin[3] + (allin[3] - allin[1]),
                            ),
                            blur_size=1,
                            colored_text=True,
                            contrast=3,
                        )
                    )
                    my_turn = True
                else:
                    current_bet = 0
        return current_bet, my_turn

    def get_num_opponents(self):
        # tally up the amount of greenbar.png there is on screen and that's the number of opponents
        greenbars = self.find_elements(
            "greenbar.png",
            search_area=(5, 181, 1343, 915),
            confidence_threshold=0.85,
            block_wide=True,
        )

        # supreme_justice
        supreme_justice = self.find_element("supreme_justice.png")
        # supreme_justice

        num_opponents = len(greenbars)
        if supreme_justice:
            num_opponents += 1
        print(f"Number of opponents: {num_opponents}")
        if self.debug:
            for greenbar in greenbars:
                print(greenbar)
        if num_opponents == 0:
            print("ERROR: Could not find any opponents")
            num_opponents = 1
        return num_opponents

    def get_pot(self, stack_size):
        pot = self.find_element("pot.png")
        pot_value = 0
        middle_pot_value = 0
        if pot:
            pot_value += self.str_to_int(
                self.ocr_text_from_image(
                    (pot[2], pot[1], pot[2] + 75, pot[3] - 2),
                    blur_size=1,
                    invert=True,
                    contrast=1,
                )
            )
            print(f"Pot found: {pot} with value {pot_value}")
        else:
            main = self.find_element("main.png")
            if main:
                value = self.str_to_int(
                    self.ocr_text_from_image(
                        (main[2], main[1] - 5, main[2] + 75, main[3]),
                        blur_size=1,
                        invert=True,
                        contrast=2,
                    )
                )
                pot_value += value
                print(f"Main found: {main} with value {value}")
                # repeat for multiple possible side pots
                side_pots = self.find_elements("side.png")
                for confidence, side_pot in side_pots:
                    value = self.str_to_int(
                        self.ocr_text_from_image(
                            (
                                side_pot[2],
                                side_pot[1] - 5,
                                side_pot[2] + 75,
                                side_pot[3],
                            ),
                            blur_size=1,
                            invert=True,
                            contrast=2,
                        )
                    )
                    pot_value += value
                    print(
                        f"Side pot found: {side_pot} with value {value} (Confidence: {confidence})"
                    )
            else:
                print("Could not find pot or main")
                pot_value = 0
        print(f"Pot value: {pot_value}")
        middle_pot_value = pot_value

        popups = [
            "allinpopup.png",
            "betpopup.png",
            "bigpopup.png",
            "smallpopup.png",
            "raisepopup.png",
            "callpopup.png",
            "postpopup.png",
            "popup.png",
        ]
        for popup in popups:
            popup_locations = self.find_elements(
                popup, search_area=(233, 312, 1148, 724), confidence_threshold=0.85
            )
            if popup_locations:
                for confidence, popup_location in popup_locations:
                    center_x = (popup_location[0] + popup_location[2]) // 2
                    popup_value = self.str_to_int(
                        self.ocr_text_from_image(
                            (
                                center_x - 40,
                                popup_location[3],
                                center_x + 40,
                                popup_location[3] + 20,
                            ),
                            blur_size=1,
                            invert=True,
                            contrast=3,
                            erode=True,
                        )
                    )
                    if self.debug:
                        print(
                            f"Popup found: {popup} at {popup_location} with value {popup_value} (Confidence: {confidence})"
                        )
                    if popup_value > stack_size:
                        print(
                            "Popup value greater than stack size, cannot win the extra amt, setting to stack size"
                        )
                        popup_value = stack_size
                    pot_value += popup_value
                print(f"New pot value after looking for {popup}: {pot_value}")
        print(f"Middle pot value: {middle_pot_value}")
        return pot_value, middle_pot_value

    def get_stack_size(self):
        username = self.find_element("usernameaction.png")
        if not username:
            username = self.find_element("usernamewhite.png")
        if not username:
            username = self.find_element("usernamegray.png")
        if not username:
            print("ERROR: Could not find username")
            stack_size = 9999999
        else:
            # show the username
            # if self.debug:
            #     cv2.imshow('username', np.array(self.screenshot.crop(username)))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            stack_location = (
                username[2] - 20,
                username[1] - 27,
                username[2] + 55,
                username[1] - 6,
            )

            stack_size = self.str_to_int(
                self.ocr_text_from_image(
                    stack_location, blur_size=1, invert=True, contrast=2
                )
            )
            print(f"Stack size: {stack_size}")
        return stack_size

    def get_cards(self):
        holecard1 = self.get_card_value_and_suit(
            self.holecard1_value_location, self.holecard1_suit_location, True
        )
        holecard2 = self.get_card_value_and_suit(
            self.holecard2_value_location, self.holecard2_suit_location, True
        )
        i = 1
        board_cards = []
        for card_value_location, card_suit_location in self.board_card_locations:
            card = self.get_card_value_and_suit(
                card_value_location, card_suit_location, False, board_card_number=i
            )
            if card[0] == 0:
                print(f"Board card {i} not visible, no need to check the rest")
                if 1 < i < 4:
                    raise ValueError(
                        "Board card 1 visible but board card 2 or 3 not visible"
                    )
                break
            board_cards.append(card)
            i += 1
        print(f"Hole card 1: {holecard1}")
        print(f"Hole card 2: {holecard2}")
        print(f"Board cards: {board_cards}")
        return board_cards, holecard1, holecard2

    def wait_random_time(self, decision, stack_size, pot_value, game_stage):
        if decision == "fold":
            return
        peak2_weight = 5
        if decision.startswith("bet"):
            bet_amount = int(decision.split()[1])
            if bet_amount >= pot_value:
                peak2_weight -= 2
            if game_stage == 0:
                peak2_weight += 3

        samples = 1000
        peak1 = np.random.normal(loc=0, scale=0.2, size=int(samples / 2))
        peak2 = np.random.normal(loc=3, scale=1, size=int(samples / peak2_weight))
        bimodal = np.concatenate([peak1, peak2])
        density = gaussian_kde(bimodal)

        xs = np.linspace(-10, 10, 200)

        density.covariance_factor = lambda: 0.25
        density._compute_covariance()
        ys = density(xs)
        wait_time = np.random.choice(xs, p=ys / np.sum(ys))
        print(f"Waiting for {max(wait_time, 0)} seconds...")
        pyautogui.sleep(max(wait_time, 0))


if __name__ == "__main__":
    main()
