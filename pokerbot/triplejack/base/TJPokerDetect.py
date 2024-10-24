# lazy code for now
import json
import math
import re
from typing import Union, List, Tuple

import pytesseract
from scipy import signal
from scipy.ndimage import convolve, binary_erosion

from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike
import cv2
import numpy as np

from treys import Card

from ...all.utils import *

import os
import time

class TJPopupTypes:
    BASE = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALLIN = 5
    POST = 6
    BIG = 7
    SMALL = 8
    FOLD = 9

    def to_str(self, val: int) -> str:
        if val == self.BASE:
            return "BASE"
        elif val == self.CHECK:
            return "CHECK"
        elif val == self.CALL:
            return "CALL"
        elif val == self.BET:
            return "BET"
        elif val == self.RAISE:
            return "RAISE"
        elif val == self.ALLIN:
            return "ALLIN"
        elif val == self.POST:
            return "POST"
        elif val == self.BIG:
            return "BIG"
        elif val == self.SMALL:
            return "SMALL"
        else:
            return "UNKNOWN"

    def from_str(self, val: str) -> int:
        if val == "BASE":
            return self.BASE
        elif val == "CHECK":
            return self.CHECK
        elif val == "CALL":
            return self.CALL
        elif val == "BET":
            return self.BET
        elif val == "RAISE":
            return self.RAISE
        elif val == "ALLIN":
            return self.ALLIN
        elif val == "POST":
            return self.POST
        elif val == "BIG":
            return self.BIG
        elif val == "SMALL":
            return self.SMALL
        else:
            return -1


class TJPokerDetect(PokerImgDetect, PokerDetection):

    def __init__(self, username: str) -> None:
        super().__init__(
            opts=PokerImgOpts(
                folder_path="pokerbot/triplejack/base/imgs",
                sit_button=("sit.png", False),
                community_hearts=("heart1.png", True),
                community_diamonds=("diamond1.png", True),
                community_clubs=("club1.png", True),
                community_spades=("spade1.png", True),
                hole_hearts=("hole_heart1.png", True),
                hole_diamonds=("hole_diamond2.png", True),
                hole_clubs=("hole_club2.png", True),
                hole_spades=("hole_spade2.png", True),
                check_button=("checkbutton.png", False),
                call_button=("callbutton.png", False),
                bet_button=("betbutton.png", False),
                fold_button=("foldbutton.png", False),
                raise_button=("raisebutton.png", False),
                allin_button=("allinbutton.png", False),
            )
        )

        self.POT_BYTES = None
        self.MAIN_POT_BYTES = None
        self.SIDE_POT_BYTES = None

        # popups
        self.CHECK_POPUP_BYTES = None
        self.CALL_POPUP_BYTES = None
        self.BET_POPUP_BYTES = None
        self.RAISE_POPUP_BYTES = None
        self.ALLIN_POPUP_BYTES = None
        self.POST_POPUP_BYTES = None
        self.BASE_POPUP_BYTES = None
        self.BIG_POPUP_BYTES = None
        self.SMALL_POPUP_BYTES = None
        self.FOLD_POPUP_BYTES = None

        self.PLUS_BUTTON_BYTES = None

        self.POPUP_BYTES = []
        self.__POPUP_DICT = {}

        self.name_loc = None
        self.username = username

        self.saved_small_blind = None
        self.saved_big_blind = None

    def load_images(self):
        super().load_images()

        self.POT_BYTES = self.load_image("pot.png")
        self.MAIN_POT_BYTES = self.load_image("mainpot.png")
        self.SIDE_POT_BYTES = self.load_image("sidepot.png")

        self.BASE_POPUP_BYTES = self.load_image("basepopup.png")
        self.CHECK_POPUP_BYTES = self.load_image("checkpopup.png")
        self.CALL_POPUP_BYTES = self.load_image("callpopup.png")
        self.BET_POPUP_BYTES = self.load_image("betpopup.png")
        self.RAISE_POPUP_BYTES = self.load_image("raisepopup.png")
        self.ALLIN_POPUP_BYTES = self.load_image("allinpopup.png")
        self.POST_POPUP_BYTES = self.load_image("postpopup.png")
        self.BIG_POPUP_BYTES = self.load_image("bigpopup.png")
        self.SMALL_POPUP_BYTES = self.load_image("smallpopup.png")
        self.FOLD_POPUP_BYTES = self.load_image("foldpopup.png")
        self.PLUS_BUTTON_BYTES = self.load_image("plusbutton.png")

        self.POPUP_BYTES = [
            self.BASE_POPUP_BYTES,
            self.CHECK_POPUP_BYTES,
            self.CALL_POPUP_BYTES,
            self.BET_POPUP_BYTES,
            self.RAISE_POPUP_BYTES,
            self.ALLIN_POPUP_BYTES,
            self.POST_POPUP_BYTES,
            self.BIG_POPUP_BYTES,
            self.SMALL_POPUP_BYTES,
            self.FOLD_POPUP_BYTES
        ]

        self.__POPUP_DICT = {
            TJPopupTypes.BASE: self.BASE_POPUP_BYTES,
            TJPopupTypes.CHECK: self.CHECK_POPUP_BYTES,
            TJPopupTypes.CALL: self.CALL_POPUP_BYTES,
            TJPopupTypes.BET: self.BET_POPUP_BYTES,
            TJPopupTypes.RAISE: self.RAISE_POPUP_BYTES,
            TJPopupTypes.ALLIN: self.ALLIN_POPUP_BYTES,
            TJPopupTypes.POST: self.POST_POPUP_BYTES,
            TJPopupTypes.BIG: self.BIG_POPUP_BYTES,
            TJPopupTypes.SMALL: self.SMALL_POPUP_BYTES,
            TJPopupTypes.FOLD: self.FOLD_POPUP_BYTES,
        }

    def ident_near_popup(self, img, loc: tuple[int, int, int, int]):
            w, h = loc[2] - loc[0], loc[3] - loc[1]
            subsection = (
                loc[0] - w // 2,
                loc[1] + h,
                loc[2] + w // 2,
                loc[3] + h * 2,
            )

            text = self.ocr_text_from_image(img, subsection, invert=True, brightness=0.5, contrast=3, erode=True)
            
            if text == "":
                return 0 
            
            try:
                # occasional noise breaks this
                return pretty_str_to_float(text)
            except ValueError:
                return 0
            
    # due to triplejack having them all at the bottom, moved this out of generic impl for speedup.
    def __get_button_subsection(self, screenshot: cv2.typing.MatLike) -> tuple[int, int, int, int]:
        h = screenshot.shape[0]
        return (0, h - h // 5, screenshot.shape[1], h)

    def call_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.CALL_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def check_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.CHECK_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def bet_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.BET_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def plus_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.PLUS_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def fold_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.FOLD_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def raise_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.RAISE_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def allin_button(self, screenshot: cv2.typing.MatLike, threshold=0.7) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.ALLIN_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))

    def popup(self, screenshot: cv2.typing.MatLike, popup_type: int) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.__POPUP_DICT[popup_type])
    
    def find_community_suits(self, ss1: cv2.typing.MatLike, threshold=0.77, subsection: Union[tuple[int, int, int, int], None]=None) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return super().find_community_suits(ss2, threshold, subsection)

    def find_hole_suits(self, ss1: cv2.typing.MatLike, threshold=0.77, subsection: Union[tuple[int, int, int, int], None]=None) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return super().find_hole_suits(ss2, threshold, subsection)

    def set_name_loc(self, img: MatLike):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image = np.float32(image)

        # if S value is above 10, set V to 0
        hsv_image[:, :, 2] = np.where(hsv_image[:, :, 1] > 20, 0, hsv_image[:, :, 2])
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

        hsv_image = np.uint8(hsv_image)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # filter out the color [68, 68, 68], inactive player gray
        mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([83, 83, 83]))
        image[mask != 0] = [0, 0, 0]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # there's no HSV2GRAY

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary = cv2.bitwise_not(binary)

        binary = self.erase_edges(binary)

        # print(pytesseract.pytesseract.tesseract_cmd)
        # print(pytesseract.image_to_boxes(binary))
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        username = self.username

        if username in data["text"]:
            index = data["text"].index(username)
            left = data["left"][index]
            top = data["top"][index]
            right = left + data["width"][index]
            bottom = top + data["height"][index]
            self.name_loc = (left, top, right, bottom)

            # show name loc on img
            # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            raise RuntimeError("Couldn't find username")

    def stack_size(self, img: MatLike) -> int:
        if self.name_loc is None:
            self.set_name_loc(img)
            if self.name_loc is None:
                print("Name location is None")
                return 999999
        center_x = (self.name_loc[0] + self.name_loc[2]) // 2
        height = self.name_loc[3] - self.name_loc[1]
        left = center_x + 20
        top = self.name_loc[1] - int(height * 2)
        right = center_x + 140
        bottom = self.name_loc[1]
        # print(self.name_loc)

        # show cropped image
        # cv2.imshow("img", img[top:bottom, left:right])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        number = self.ocr_text_from_image(img, (left, top, right, bottom), invert=True, brightness=0.5, contrast=3, erode=False)
        return pretty_str_to_float(number)

    def middle_pot(self, img: MatLike) -> int:
        print("Getting middle pot...")
        def ident_near_pot(img, loc: tuple[int, int, int, int]):
            w, h = loc[2] - loc[0], loc[3] - loc[1]
            subsection = (
                loc[0] + w,
                loc[1] - h // 6,
                loc[2] + w * 4,
                loc[3] + h // 6,
            )
            text = self.ocr_text_from_image(img, subsection, invert=True)

            return pretty_str_to_float(text)

        h = img.shape[0]
        w = img.shape[1]
        subsection = (w // 3, h // 4, w // 3 * 2, h // 4 * 3)
    
        single_pot = self.ident_one_template(img, self.POT_BYTES, subsection=subsection)

        if single_pot is None:

            main_pot = self.ident_one_template(img, self.MAIN_POT_BYTES, subsection=subsection)

            # no pots currently visible
            if main_pot is None:
                return 0
            
            side_pots = self.template_detect(img, self.SIDE_POT_BYTES, subsection=subsection)
            if len(side_pots) == 0:
                return ident_near_pot(img, main_pot)

            sum = 0
            for pot in side_pots:
                sum += ident_near_pot(img, pot)
                
            return sum + ident_near_pot(img, main_pot)
        
        else:
            return ident_near_pot(img, single_pot)


    def total_pot(self, img: MatLike) -> int:
        return self.middle_pot(img) + sum(self.current_bets(img))

    def current_bets(self, img: MatLike) -> list[int]:
        ret = []
        print("Getting current bets...")

        # for popup in self.POPUP_BYTES:
        #     for loc in self.template_detect(img, popup, threshold=0.9):
        #         ret.append(self.ident_near_popup(img, loc))
        #
        # return ret

        subsection = (img.shape[1] // 5, (img.shape[0] * 2) // 7, img.shape[1] * 4 // 5, (6 * img.shape[0]) // 11)
        cards_to_zero_out = (425, 305, 425 + 475, 305 + 130)
        main_pot_to_zero_out = (615, 435, 615 + 100, 435 + 100)
        img = img.copy()
        img[cards_to_zero_out[1]:cards_to_zero_out[3], cards_to_zero_out[0]:cards_to_zero_out[2]] = 0
        img[main_pot_to_zero_out[1]:main_pot_to_zero_out[3], main_pot_to_zero_out[0]:main_pot_to_zero_out[2]] = 0
        img = img[subsection[1]:subsection[3], subsection[0]:subsection[2]]

        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image = np.float32(image)

        # if S value is above 10, set V to 0
        hsv_image[:, :, 2] = np.where(hsv_image[:, :, 1] > 20, 0, hsv_image[:, :, 2])
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

        # if v value is below 200, set V to 0
        # hsv_image[:, :, 2] = np.where(hsv_image[:, :, 2] < 175, 0, hsv_image[:, :, 2])
        # hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

        hsv_image = np.uint8(hsv_image)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary = cv2.bitwise_not(binary)

        binary = self.erase_edges(binary)

        # cv2.imshow("img", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)

        number_regex = re.compile(r"\d{1,3}(?:,\d{3})*(?:[kKmM])?")

        for i in range(len(data["text"])):
            text = data["text"][i]
            if len(text) > 0 and number_regex.match(text):
                # left = data["left"][i]
                # top = data["top"][i]
                # right = left + data["width"][i]
                # bottom = top + data["height"][i]
                ret.append(pretty_str_to_float(text))

        return ret
        
    def current_bet(self, img: MatLike) -> int:
        # TODO: obsolesce this method by tracking player bet and max of current_bets()
        check_button = self.check_button(img)
        if check_button is not None:
            return 0
        else:
            call_button = self.call_button(img)
            if call_button is not None:
                loc = (call_button[0] - 10, call_button[3], call_button[2] + 10, call_button[3] + (call_button[3] - call_button[1]) + 3)
                return pretty_str_to_float(self.ocr_text_from_image(img, loc, contrast=3))
            else:
                allin_button = self.allin_button(img)
                if allin_button is not None:
                    loc = (allin_button[0] - 10, allin_button[3], allin_button[2] + 10, allin_button[3] + (allin_button[3] - allin_button[1]) + 3)
                    return pretty_str_to_float(self.ocr_text_from_image(img, loc, contrast=3))
                else:
                    print("ERROR: Not my turn or couldn't find current bet")
                    return 999999


    def min_bet(self, img: MatLike) -> int:
        # when you call this, check if current_bet >= stack_size, if it is, save time by not calling this method
        bet_button = self.bet_button(img)
        if bet_button is not None:
            loc = (bet_button[0] - 10, bet_button[3], bet_button[2] + 10, bet_button[3] + (bet_button[3] - bet_button[1]) + 3)
            return pretty_str_to_float(self.ocr_text_from_image(img, loc, contrast=3))
        else:
            raise_button = self.raise_button(img)
            if raise_button is not None:
                loc = (raise_button[0] - 10, raise_button[3], raise_button[2] + 10,
                       raise_button[3] + (raise_button[3] - raise_button[1]) + 3)
                return pretty_str_to_float(self.ocr_text_from_image(img, loc, contrast=3))
            else:
                allin_button = self.allin_button(img)
                if allin_button is not None:
                    loc = (allin_button[0] - 10, allin_button[3], allin_button[2] + 10,
                           allin_button[3] + (allin_button[3] - allin_button[1]) + 3)
                    return pretty_str_to_float(self.ocr_text_from_image(img, loc, contrast=3))
                else:
                    raise RuntimeError("ERROR: Not my turn or couldn't find current bet")

    def table_players(self, img: MatLike, active=False) -> List[Tuple[Tuple[int, int, int, int], str]]:
        # convert from BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # crop bottom 20% of image off
        rgb_img = rgb_img[:int(rgb_img.shape[0] * 0.8)]

        if not active:
            filter_colors = [
                [68, 68, 68],  # inactive player gray
                [48, 192, 86],  # active player green
                [28, 43, 53],  # action player gray
                [57, 72, 82],  # action player gray brightened by glow
                [45, 60, 70],  # action player gray brightened by glow
            ]
        else:
            filter_colors = [
                [48, 192, 86],  # active player green
                [28, 43, 53],  # action player gray
                [57, 72, 82],  # action player gray brightened by glow
                [45, 60, 70],  # action player gray brightened by glow
            ]
            filter_colors.extend([[x, x, x] for x in range(235, 256)])
            filter_colors = np.array(filter_colors)

        mask = np.zeros_like(rgb_img[:, :, 0], dtype=bool)

        for color in filter_colors:
            distance_mask = np.linalg.norm(rgb_img - color, axis=-1) <= 4
            mask = mask | distance_mask

        filtered_image = np.zeros_like(rgb_img)
        filtered_image[mask] = rgb_img[mask]

        modified_img = filtered_image

        gray = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        rows, cols = binary.shape

        for row in range(rows):
            white_pixel_count = np.sum(binary[row] == 255)

            if white_pixel_count < 75:
                binary[row][binary[row] == 255] = 0

        for i in range(3):
            kernel = np.concatenate([np.ones(19), np.zeros(19)])
            kernel = np.expand_dims(kernel, 0)
            left_conv = convolve(binary == 255, kernel, mode='constant') > 0
            kernel = kernel[:, ::-1]
            right_conv = convolve(binary == 255, kernel, mode='constant') > 0
            binary = binary.astype(np.uint8)
            binary[(binary == 0) & (left_conv == True) & (right_conv == True)] = 255

        for i in range(1):
            kernel = np.array([[0, 1 / 3, 0],
                               [0, 1 / 3, 0],
                               [0, 1 / 3, 0]])
            convolved_image = convolve(binary, kernel)
            binary = np.where(convolved_image == 170, 0, binary)

        # show binary
        # cv2.imshow("img", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        lines: list[list[list[int, int, int, int]]] = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 90, threshold=75, minLineLength=200, maxLineGap=37)
        if lines is None:
            return []

        # debug_image = modified_img.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(debug_image, (x1, y1), (x2, y2), (64, 255, 64), 1)
        # cv2.imshow("img", debug_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # filter out lines of length more than 260
        lines = list(filter(lambda x: math.sqrt((x[0][0] - x[0][2]) ** 2 + (x[0][1] - x[0][3]) ** 2) < 260, lines))
        # filter out lines whose y1 and y2 arent within 5 pixels
        lines = list(filter(lambda x: abs(x[0][1] - x[0][3]) < 8, lines))
        # average y values
        for line in lines:
            line[0][1] = (line[0][1] + line[0][3]) // 2
            line[0][3] = line[0][1]

        # debug_image = modified_img.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(debug_image, (x1, y1), (x2, y2), (64, 255, 64), 1)
        # cv2.imshow("img", debug_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        def distance(p1, p2):
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        def merge_lines(line1, line2):
            x1 = (line1[0] + line2[0]) / 2
            y1 = (line1[1] + line2[1]) / 2
            x2 = (line1[2] + line2[2]) / 2
            y2 = (line1[3] + line2[3]) / 2
            return [x1, y1, x2, y2]

        def clean_lines(lines, threshold, merge=False):
            i = 0
            while i < len(lines):
                j = i + 1
                while j < len(lines):
                    line1 = lines[i][0]
                    line2 = lines[j][0]
                    distances = [distance(line1[:2], line2[:2]), distance(line1[2:], line2[2:]),
                                 distance(line1[:2], line2[2:]), distance(line1[2:], line2[:2])]
                    if sum([d < threshold for d in distances]) > 1:
                        if merge:
                            merged_line = merge_lines(line1, line2)
                            lines[i][0] = merged_line
                        lines.pop(j)
                        j = i + 1
                        continue
                    j += 1
                i += 1

        upper_threshold = 35
        lower_threshold = 5
        clean_lines(lines, lower_threshold)
        clean_lines(lines, upper_threshold, merge=True)

        # debug_image = modified_img.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(debug_image, (x1, y1), (x2, y2), (64, 255, 64), 1)
        # cv2.imshow("img", debug_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        filter_color = [68, 68, 68]
        mask = cv2.inRange(rgb_img, np.array(filter_color) - 10, np.array(filter_color) + 10)
        rgb_img[mask != 0] = [0, 0, 0]
        output_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        players = []
        # show lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            top = y1 - 15
            bottom = y1 + 15
            left = x1
            right = x2
            if active:
                name = "OMITTED FOR TIME"
            else:
                name = self.ocr_text_from_image(output_image, (left, top, right, bottom), invert=True, brightness=1, contrast=1.5, card_chars=False, scale=50)
            players.append(((left, top, right, bottom), name))
        return players

    def active_players(self, img: MatLike) -> list:
        print("Getting active players...")
        return self.table_players(img, active=True)

    def big_blind(self, img: MatLike) -> int:
        if self.saved_big_blind is None:
            big_blind_popup = self.ident_one_template(img, self.BIG_POPUP_BYTES)
            if big_blind_popup is None:
                return -1
            self.saved_big_blind = self.ident_near_popup(img, big_blind_popup)
        else:
            return self.saved_big_blind
        if big_blind_popup is None:
            return -1
        
        return self.ident_near_popup(img, big_blind_popup)
    
    def small_blind(self, img: MatLike) -> int:
        if self.saved_small_blind is None:
            small_blind_popup = self.ident_one_template(img, self.SMALL_POPUP_BYTES)
            if small_blind_popup is None:
                return -1
            self.saved_small_blind = self.ident_near_popup(img, small_blind_popup)
        else:
            return self.saved_small_blind
        small_blind_popup = self.ident_one_template(img, self.SMALL_POPUP_BYTES)
        if small_blind_popup is None:
            return -1
        
        return self.ident_near_popup(img, small_blind_popup)
        

    def get_full_cards(self, img: MatLike, hole=False, subsection: Union[tuple[int, int, int, int], None]=None) -> list[tuple[Card, tuple[int, int, int, int]]]:
        if hole:
            suits = self.find_hole_suits(img, subsection=subsection)
        else:
            suits = self.find_community_suits(img, subsection=subsection)
        ret = []

        tuples_list = []
        for key, array in suits.items():
            for coords in array:
                tuples_list.append(tuple(coords) + (key,))

        sorted_tuples = sorted(tuples_list)

        i = 0
        angles = {1: -5, 2: -3, 3: 0, 4: 3, 5: 5}

        for o in sorted_tuples:
            key = o[4]
            loc = o
            w, h = loc[2] - loc[0], loc[3] - loc[1]

            if hole:
                text_area = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 5,
                    loc[2] + w // 5,
                    loc[3] - h,
                )
            else:
                text_area = (
                    loc[0] - w // 5,
                    loc[1] - h - h // 7 if not key == "hearts" else loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3] - h,
                )

            i += 1

            text = self.ocr_text_from_image(img,
                                            text_area,
                                            psm=7,
                                            contrast=1.5,
                                            black_text=True,
                                            rotation_angle=2 if hole else angles[i])

            loc = text_area

            full_card_str = f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}"

            # img1 = cv2.rectangle(img.copy(), (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)
            # cv2.putText(img1, full_card_str, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #
            # cv2.imshow("img", img1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if text == "":
                cv2.imwrite(f"error-{int(time.time())}.png", img)
                cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                cv2.imwrite(f"error-{int(time.time())}_2.png", img)
                raise ValueError("OCR failed to find card's text")

            try:
                ret.append((Card.new(full_card_str), loc))
            except KeyError as e:
                cv2.imwrite(f"error-{int(time.time())}.png", img)
                cv2.putText(img, full_card_str, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                cv2.imwrite(f"error-{int(time.time())}_2.png", img)
                raise e


        # sort ret by x position (want left to right)
        ret.sort(key=lambda x: x[1][0])

        return ret

    # TODO update these functions to use the new ident_template subsection
    def community_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the middle 4th of the screen

        h = img.shape[0]
        w = img.shape[1]
        # img = img[h // 4 : h // 4 * 3, w // 4: w // 4 * 3, :]
        subsection = (w // 4, h // 4, w // 4 * 3, h // 5 * 3)
        ret = self.get_full_cards(img, subsection=subsection)

        # shift the y position of the cards up 1/2th of the height of the image
        # return list(map(lambda x: (x[0], (x[1][0] + w // 4, x[1][1] + h // 4, x[1][2] + w // 4, x[1][3] + h // 4)), ret))
        return ret



    def community_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.community_cards_and_locs(img)))


    def hole_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the bottom 4th of the screen
        h = img.shape[0]
        w = img.shape[1]
        # img = img[img.shape[0] // 4 * 3 :, : w // 4, :]

        subsection = (0, h // 4 * 3, w // 4, h)
        ret = self.get_full_cards(img, hole=True, subsection=subsection)

    
        # shift the y position of the cards up 1/2th of the height of the image
        # return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4 * 3, x[1][2], x[1][3] + h // 4 * 3)), ret))
        return ret

    def hole_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.hole_cards_and_locs(img)))
    




       
def report_info(detector: TJPokerDetect, ss: Union[str, cv2.typing.MatLike]):

    if isinstance(ss, str):
        img = cv2.imread(ss, cv2.IMREAD_COLOR)
       
    else:
         img = ss
       
    img2 = img.copy()


    now = time.time()


    info = detector.community_cards_and_locs(img)

    for card, loc in info:
        Card.print_pretty_card(card)
        cv2.putText(img2, Card.int_to_str(card), (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img2, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)


    info1 = detector.hole_cards_and_locs(img)
    
    for card, loc in info1:
        Card.print_pretty_card(card)
        cv2.putText(img2, Card.int_to_str(card), (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img2, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    sit_locs = detector.sit_buttons(img)
    print(sit_locs)

    for loc in sit_locs:

        cv2.rectangle(img2, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    call_loc = detector.call_button(img)
    if call_loc is not None:
        print("call button found")
        print(call_loc)
        cv2.rectangle(img2, (call_loc[0], call_loc[1]), (call_loc[2], call_loc[3]), (0, 255, 0), 2)
    else:
        print("no call button found")


    check_loc = detector.check_button(img)
    if check_loc is not None:
        print("check button found")
        cv2.rectangle(img2, (check_loc[0], check_loc[1]), (check_loc[2], check_loc[3]), (0, 255, 0), 2)
    else:
        print("no check button found")

    bet_loc = detector.bet_button(img)
    if bet_loc is not None:
        print("bet button found")
        cv2.rectangle(img2, (bet_loc[0], bet_loc[1]), (bet_loc[2], bet_loc[3]), (0, 255, 0), 2)
    else:
        print("no bet button found")

    fold_loc = detector.fold_button(img)
    if fold_loc is not None:
        print("fold button found")
        cv2.rectangle(img2, (fold_loc[0], fold_loc[1]), (fold_loc[2], fold_loc[3]), (0, 255, 0), 2)
    else:
        print("no fold button found")

    raise_loc = detector.raise_button(img)
    if raise_loc is not None:
        cv2.rectangle(img2, (raise_loc[0], raise_loc[1]), (raise_loc[2], raise_loc[3]), (0, 255, 0), 2)
    else:
        print("no raise button found")

    allin_loc = detector.allin_button(img)
    if allin_loc is not None:
        print("allin button found")
        cv2.rectangle(img2, (allin_loc[0], allin_loc[1]), (allin_loc[2], allin_loc[3]), (0, 255, 0), 2)
    else:
        print("no allin button found")

    mid_pot = 0
    tot_pot = 0
    current_bets = []
    mid_pot = detector.middle_pot(img)
    current_bets = detector.current_bets(img)
    
    tot_pot = mid_pot + sum(current_bets)
    current_bet = max(current_bets) if len(current_bets) > 0 else 0

    active_players = detector.active_players(img)

    table_players = detector.table_players(img)


    filename = ss if isinstance(ss, str) else "current image"

    

    cv2.imshow(f"{filename} ({img.shape[0]}x{img.shape[1]}) | Stage: {PokerStages.to_str(cards_to_stage(info))} | mid pot: {mid_pot} | tot pot: {tot_pot} | current bets: {current_bets} (facing: {current_bet}) | Active players: {active_players} | Table players: {table_players} | Took: {int((time.time() - now) * 1000) / 1000} sec", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect = TJPokerDetect()
    detect.load_images()

    folder = "pokerbot/triplejack/base/tests"
    # for all files in a directory, run the report_info function
    files = os.listdir(folder)
    # files = [
    #     # "heck.png"
    #     "/home/generel/Documents/code/python/poker/LetsGoGambling/triplejack/new/base/tests/midrun/test-1720622523.png"
    # ]
    files = sorted(files)
    files.reverse()
    print(folder, files)
    for filename in files:
        if filename.endswith(".png"):
            path =  os.path.join(folder, filename)
            print("running report_info on",path)

           
            report_info(detect, path)
          
        
    

       

