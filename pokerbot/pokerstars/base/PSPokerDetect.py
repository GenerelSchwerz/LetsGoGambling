# lazy code for now
import json
import math
from typing import Union, List, Tuple

import pytesseract

from ...all.windows import AWindowManager

from ...abstract.pokerDetection import Player
from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike
import cv2
import numpy as np

from treys import Card

from ...all.utils import *


def pretty_str_to_int(s: str) -> int:
    try:
        return int(s.replace(",", "").replace("$", "").replace(" ", "").replace(".", ""))
    except ValueError:
        print(f"Could not convert number to int: {s}")
        return 0

class PSPokerDetect(PokerImgDetect, PokerDetection):

    def __init__(self, username: str = None) -> None:
        super().__init__(
            opts=PokerImgOpts(
                folder_path="pokerbot/pokerstars/base/imgs",
                sit_button=("sit.png", False),
                community_hearts=("community_heart.png", True),
                community_diamonds=("community_diamond.png", True),
                community_clubs=("community_club.png", True),
                community_spades=("community_spade.png", True),
                hole_hearts=("hole_heart.png", True),
                hole_diamonds=("hole_diamond.png", True),
                hole_clubs=("hole_club.png", True),
                hole_spades=("hole_spade.png", True),
                check_button=("check.png", False),
                call_button=("call.png", False),
                bet_button=("bet.png", False),
                fold_button=("fold.png", False),
                raise_button=("raise.png", False),
                allin_button=("max.png", False),
            )
        )

        self.POT_BYTES = None
        self.MAIN_POT_BYTES = None
        self.SIDE_POT_BYTES = None

   
        self.PLUS_BUTTON_BYTES = None

        # popups
        self.POPUP_LEFT_BYTES = None
        self.POPUP_RIGHT_BYTES = None

        self.PLAYER_LEFT_BYTES = None
        self.PLAYER_RIGHT_BYTES = None

        self.POPUP_BYTES = []
        self.PLAYER_BYTES = []

        self.name_loc = None
        self.username = username

    def load_images(self):
        super().load_images()

        self.POT_BYTES = self.load_image("pot.png")
        self.MAIN_POT_BYTES = self.load_image("mainpot.png")
        self.SIDE_POT_BYTES = self.load_image("sidepot.png")

        # popups
        self.POPUP_LEFT_BYTES = self.load_image("popup_left1.png")
        self.POPUP_RIGHT_BYTES = self.load_image("popup_right1.png")
        self.POPUP_CENTER_BYTES = self.load_image("popup_center.png")

        self.PLAYER_LEFT_BYTES = self.load_image("player_left_active.png")
        self.PLAYER_RIGHT_BYTES = self.load_image("player_right_active.png")

        self.POPUP_BYTES = [
            self.POPUP_LEFT_BYTES,
            self.POPUP_RIGHT_BYTES,
            self.POPUP_CENTER_BYTES,
        ]

        self.PLAYER_BYTES = [self.PLAYER_LEFT_BYTES, self.PLAYER_RIGHT_BYTES]

    def ident_near_popup(
        self, img, info_to_right: bool, loc: tuple[int, int, int, int]
    ):
        w, h = loc[2] - loc[0], loc[3] - loc[1]

        subsection = (
            loc[0] + (w if info_to_right else -w * 20),
            loc[1],
            loc[2] + (w * 20 if info_to_right else -w),
            loc[3],
        )

        print(w, h)
        cv2.imshow("raw", img[loc[1] : loc[3], loc[0] : loc[2]])
        cv2.imshow(
            f"img | info to right: {info_to_right}",
            img[subsection[1] : subsection[3], subsection[0] : subsection[2]],
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        text = self.ocr_text_from_image(
            img, subsection, invert=True, brightness=0.5, contrast=3, erode=True
        )

        if text == "":
            return 0

        try:
            # occasional noise breaks this
            return pretty_str_to_int(text)
        except ValueError:
            return 0

    # due to triplejack having them all at the bottom, moved this out of generic impl for speedup.
    def __get_button_subsection(
        self, screenshot: cv2.typing.MatLike
    ) -> tuple[int, int, int, int]:
        h = screenshot.shape[0]
        return (screenshot.shape[1] // 2, h - h // 4, screenshot.shape[1], h)

    def call_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.CALL_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def check_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.CHECK_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def bet_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.BET_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def plus_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.PLUS_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def fold_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.FOLD_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def raise_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.RAISE_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def allin_button(
        self, screenshot: cv2.typing.MatLike, threshold=0.8
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(
            screenshot,
            self.ALLIN_BUTTON_BYTES,
            threshold,
            self.__get_button_subsection(screenshot),
        )

    def popup(
        self, screenshot: cv2.typing.MatLike, popup_type: int
    ) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.__POPUP_DICT[popup_type])

    def find_community_suits(
        self,
        ss1: cv2.typing.MatLike,
        threshold=0.77,
        subsection: Union[tuple[int, int, int, int], None] = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        # _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return super().find_community_suits(ss2, threshold, subsection)

    def find_hole_suits(
        self,
        ss1: cv2.typing.MatLike,
        threshold=0.77,
        subsection: Union[tuple[int, int, int, int], None] = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        # _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        else:
            raise RuntimeError("Couldn't find username")

    def stack_size(self, img: MatLike) -> int:
        """
        Due to PokerStars having a scalable screen and always sitting us as the bottom,
        We are able to hardcode the hell out of this.
        """

        w, h = img.shape[1], img.shape[0]
        subsection = (w // 16 * 6, h // 32 * 24, w // 16 * 9, h // 32 * 25)
        text = self.ocr_text_from_image(
            img, subsection, invert=True, brightness=0.5, contrast=3, erode=True
        )
        return pretty_str_to_int(text)

    def middle_pot(self, img: MatLike) -> int:

        def ident_near_pot(img, loc: tuple[int, int, int, int]):
            w, h = loc[2] - loc[0], loc[3] - loc[1]
            subsection = (
                loc[0] + w,
                loc[1] - h // 6,
                loc[2] + w * 4,
                loc[3] + h // 6,
            )
            text = self.ocr_text_from_image(img, subsection, invert=True)

            return pretty_str_to_int(text)

        h = img.shape[0]
        w = img.shape[1]
        subsection = (w // 3, h // 4, w // 3 * 2, h // 4 * 3)
        #      fullimg = fullimg[subsection[1]:subsection[3], subsection[0]:subsection[2]]

        # cv2.imshow(
        #     "img", img[subsection[1] : subsection[3], subsection[0] : subsection[2]]
        # )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        single_pot = self.ident_one_template(
            img, self.POT_BYTES, subsection=subsection, threshold=0.5
        )

        if single_pot is None:

            main_pot = self.ident_one_template(
                img, self.MAIN_POT_BYTES, subsection=subsection
            )

            # no pots currently visible
            if main_pot is None:
                return 0

            side_pots = self.template_detect(
                img, self.SIDE_POT_BYTES, subsection=subsection
            )
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

    def find_popup_info(
        self, img: MatLike
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        ret = []

        def work(subsection):
            # cv2.imshow(
            #     "work img",
            #     img[subsection[1] : subsection[3], subsection[0] : subsection[2]],
            # )
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            text = self.ocr_text_from_image(
                img, subsection, invert=True, brightness=0.5, contrast=3, erode=True
            )

            if text == "":
                return 0

            try:
                # occasional noise breaks this
                return pretty_str_to_int(text)
            except ValueError:
                return 0

        for loc in self.template_detect(img, self.POPUP_CENTER_BYTES, threshold=0.9):
            w, h = loc[2] - loc[0], loc[3] - loc[1]
            subsection = (loc[0] - w * 5, loc[1] - 5, loc[2] + w * 5, loc[3] + 5)

            check_area = img[
                subsection[1] : subsection[3], subsection[0] : subsection[2]
            ]
            # cv2.imshow("check_area", check_area)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            left_ret = self.template_detect(
                check_area, self.POPUP_LEFT_BYTES, threshold=0.9
            )

            if len(left_ret) != 0:
                left_ret = left_ret[0]
                print("found left popup (wanted info to the right)")
                # make a new subsection between the left popup and the center popup
                subsection1 = (
                    subsection[0] + left_ret[0],
                    subsection[1] + left_ret[1],
                    loc[2] - w,
                    subsection[1] + left_ret[3],
                )
                # img2 = cv2.rectangle(
                #     img.copy(),
                #     (subsection1[0], subsection1[1]),
                #     (subsection1[2], subsection1[3]),
                #     (0, 255, 0),
                #     2,
                # )
                # cv2.imshow("img", img2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                ret.append((work(subsection1), subsection1))

            else:
                right_ret = self.template_detect(
                    check_area, self.POPUP_RIGHT_BYTES, threshold=0.9
                )
                if len(right_ret) != 0:
                    # correct off
                    right_ret = right_ret[0]

                    print("found right popup (wanted info to the left)")
                    # make a new subsection between the right popup and the center popup
                    subsection1 = (
                        loc[0] + w,
                        subsection[1] + right_ret[1],
                        subsection[0] + right_ret[2],
                        subsection[1] + right_ret[3],
                    )

                    # img2 = cv2.rectangle(
                    #     img.copy(),
                    #     (subsection1[0], subsection1[1]),
                    #     (subsection1[2], subsection1[3]),
                    #     (0, 255, 0),
                    #     2,
                    # )
                    # cv2.imshow("img", img2)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    ret.append((work(subsection1), subsection1))

                else:
                    continue
        return ret

    def current_bets(self, img: MatLike) -> list[int]:
        return list(map(lambda x: x[0], self.find_popup_info(img)))

    def current_bet(self, img: MatLike) -> int:
        # TODO: obsolesce this method by tracking player bet and max of current_bets()
        return max(self.current_bets(img))

    def min_bet(self, img: MatLike) -> int:
        # when you call this, check if current_bet >= stack_size, if it is, save time by not calling this method
        bet_button = self.bet_button(img)
        if bet_button is not None:
            w, h = bet_button[2] - bet_button[0], bet_button[3] - bet_button[1]
            loc = (
                bet_button[0] - 10,
                bet_button[3],
                bet_button[2] + 10,
                bet_button[3] + (bet_button[3] - bet_button[1]) * 2
            )

            return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
        elif (raise_button := self.raise_button(img)) is not None:
            
        
            loc = (
                raise_button[0] - 10,
                raise_button[3],
                raise_button[2] + 10,
                raise_button[3] + (raise_button[3] - raise_button[1]) * 2,
            )
            return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
        else:   
            raise RuntimeError("Not my turn or couldn't find current bet")

    def table_players(
        self, img: MatLike
    ) -> List[Tuple[Player, Tuple[int, int, int, int]]]:
        players = []

        sections = []
        img2 = img.copy()
        for loc in self.template_detect(img, self.PLAYER_LEFT_BYTES, threshold=0.9):
            w, h = loc[2] - loc[0], loc[3] - loc[1]

            # player info is to the right
            sections.append( (
                loc[0],
                loc[1] - 5,
                loc[2] + w * 5,
                loc[3] + 5,
            )
          )
    
        for loc in self.template_detect(img, self.PLAYER_RIGHT_BYTES, threshold=0.9):
            w, h = loc[2] - loc[0], loc[3] - loc[1]
            
            # player info is to the left
            sections.append((
                loc[0] - w * 5 - w // 2, # lazy fix
                loc[1] - 5,
                loc[2],
                loc[3] + 5,
            )) 
      
        for section in sections:
            w, h = section[2] - section[0], section[3] - section[1]

            tot_bright = np.sum(img2[section[1] : section[3], section[0] : section[2]]) // (3 * w * h)

            print(tot_bright)
            active = tot_bright > 30

            # name in top half of section
            subsection = (section[0], section[1], section[2], section[1] + h // 2)

            name = self.ocr_text_from_image(
                img, subsection, invert=True, brightness=0.5, contrast=3, erode=True, allowed_chars=False
            )

            # stack size in bottom half of section
            subsection = (section[0], section[1] + h // 2, section[2], section[3])
            stack_size = self.ocr_text_from_image(
                img, subsection, invert=True, brightness=0.5, contrast=3, erode=False, similarity_factor=False
            )

            if stack_size == "":
                # only happens when a player is disconnected or sitting out, so we'll essentially ignore them.
                continue
            else:
                stack_size = pretty_str_to_int(stack_size)

            players.append((Player(name, stack_size, active=active), section))

            

         

        # cv2.imshow("img", img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return players

    def active_players(self, img: MatLike) -> list:
        players = self.table_players(img)

      

        return players

    def big_blind(self, img: MatLike) -> int:
        big_blind_popup = self.ident_one_template(img, self.BIG_POPUP_BYTES)
        if big_blind_popup is None:
            return -1

        return self.ident_near_popup(img, big_blind_popup)

    def small_blind(self, img: MatLike) -> int:
        small_blind_popup = self.ident_one_template(img, self.SMALL_POPUP_BYTES)
        if small_blind_popup is None:
            return -1

        return self.ident_near_popup(img, small_blind_popup)

    def get_full_cards(
        self,
        img: MatLike,
        hole=False,
        subsection: Union[tuple[int, int, int, int], None] = None,
    ) -> list[tuple[Card, tuple[int, int, int, int]]]:
        if hole:
            suits = self.find_hole_suits(img, subsection=subsection)
        else:
            suits = self.find_community_suits(img, subsection=subsection)
          

        ret = []

        for key, locs in suits.items():
            for loc in locs:
                w, h = loc[2] - loc[0], loc[3] - loc[1]

                if hole:
                    text_area = (
                        loc[0] - w ,
                        loc[1] - 3 * h,
                        loc[2] + w,
                        loc[3] - h,
                    )
                else:
                    # shift text area down 1/2th of the height of the image and over 1/2th of the width of the image
                    text_area = (
                        loc[0] + w // 2,
                        loc[1] + h,
                        loc[2] + 3 * w // 2,
                        loc[3] + 4 * h // 2 + h // 4,
                    )


                # cv2.imshow(
                #     "img",
                #     img[text_area[1] : text_area[3], text_area[0] : text_area[2]],
                # )
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                text = self.ocr_text_from_image(
                    img,
                    text_area,
                    psm=7,
                    contrast=1.5,
                    invert=True,
                    rotation_angle=0,
                )

                loc = text_area

                full_card_str = f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}"

                # img1 = cv2.rectangle(img.copy(), (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)
                # cv2.putText(img1, full_card_str, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.imshow("img", img1)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if text == "":
                    cv2.imwrite(f"error-{int(time.time())}.png", img)
                    cv2.rectangle(
                        img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2
                    )
                    cv2.imwrite(f"error-{int(time.time())}_2.png", img)
                    raise ValueError("OCR failed to find card's text")

                try:
                    ret.append((Card.new(full_card_str), loc))
                except KeyError as e:
                    cv2.imwrite(f"error-{int(time.time())}.png", img)
                    cv2.putText(
                        img,
                        full_card_str,
                        (loc[0], loc[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.rectangle(
                        img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2
                    )
                    cv2.imwrite(f"error-{int(time.time())}_2.png", img)
                    raise e

        # sort ret by x position (want left to right)
        ret.sort(key=lambda x: x[1][0])

        return ret

    # TODO update these functions to use the new ident_template subsection
    def community_cards_and_locs(
        self, img: MatLike
    ) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the middle 4th of the screen

        h = img.shape[0]
        w = img.shape[1]
        # img = img[h // 4 : h // 4 * 3, w // 4: w // 4 * 3, :]
        subsection = (w // 4, h // 3, w // 4 * 3, h // 2)

        # cv2.imshow("img", img[subsection[1] : subsection[3], subsection[0] : subsection[2]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ret = self.get_full_cards(img, subsection=subsection)

        # shift the y position of the cards up 1/2th of the height of the image
        # return list(map(lambda x: (x[0], (x[1][0] + w // 4, x[1][1] + h // 4, x[1][2] + w // 4, x[1][3] + h // 4)), ret))
        return ret

    def community_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.community_cards_and_locs(img)))

    def hole_cards_and_locs(
        self, img: MatLike
    ) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the bottom 4th of the screen
        h = img.shape[0]
        w = img.shape[1]
        # img = img[img.shape[0] // 4 * 3 :, : w // 4, :]

        subsection = (w // 3, h // 4 * 2, 2 * w // 3, h // 4 * 3)

        # cv2.imshow("img", img[subsection[1] : subsection[3], subsection[0] : subsection[2]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ret = self.get_full_cards(img, hole=True, subsection=subsection)

        # shift the y position of the cards up 1/2th of the height of the image
        # return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4 * 3, x[1][2], x[1][3] + h // 4 * 3)), ret))
        return ret

    def hole_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.hole_cards_and_locs(img)))


def report_info(detector: PSPokerDetect, ss: Union[str, cv2.typing.MatLike]):

    if isinstance(ss, str):
        img = cv2.imread(ss, cv2.IMREAD_COLOR)

    else:
        img = ss

    img2 = img.copy()

    now = time.time()


    info = []

    info = detector.community_cards_and_locs(img)

    print(info)
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
    print("sit locs", sit_locs)

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
    current_bet = 0
    min_bet = 0
    active_players = []
    table_players = []

    mid_pot = detector.middle_pot(img)

    print("midpot", mid_pot)
    current_bets = detector.current_bets(img)

    print("current bets", current_bets)

    tot_pot = mid_pot + sum(current_bets)
    current_bet = max(current_bets) if len(current_bets) > 0 else 0

    active_players = detector.active_players(img)

    table_players = detector.table_players(img)

    print("players: \n",table_players, len(table_players))
    

    for player, loc in table_players:
        color = (0, 255, 0) if player.active else (0, 0, 255)
        cv2.putText(img2, player.name, (loc[0], loc[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img2, str(player.stack), (loc[0], loc[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(img2, (loc[0], loc[1]), (loc[2], loc[3]), color, 2)

    min_bet = detector.min_bet(img)

    print("min bet", min_bet)

    filename = ss if isinstance(ss, str) else "current image"

    cv2.imshow(
        f"{filename} ({img.shape[0]}x{img.shape[1]}) | Stage: {PokerStages.to_str(cards_to_stage(info))} | mid pot: {mid_pot} | tot pot: {tot_pot} | current bets: {current_bets} (facing: {current_bet}) | Active players: {active_players} | Table players: {table_players} | Took: {int((time.time() - now) * 1000) / 1000} sec",
        img2,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect = PSPokerDetect()
    detect.load_images()
    import os
    import time

    folder = "pokerbot/pokerstars/base/tests"
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
        if filename.endswith(".png") and "cards1" in filename:
            path = os.path.join(folder, filename)
            print("running report_info on", path)

            report_info(detect, path)
