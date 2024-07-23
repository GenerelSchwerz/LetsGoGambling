# lazy code for now


import math
from typing import Union
from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike
import cv2
import numpy as np

from treys import Card

from .utils import *


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

    def __init__(self) -> None:
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
                allin_button=("allinbutton.png", False)
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

        self.POPUP_BYTES = []
        self.__POPUP_DICT = {}
    

        self.seat_loc = None

    def load_images(self):
        super().load_images()

        self.POT_BYTES = self.load_image("pot.png")
        self.MAIN_POT_BYTES = self.load_image("mainpot.png")
        self.SIDE_POT_BYTES = self.load_image("sidepot.png")

        self.BASE_POPUP_BYTES = self.load_image("basepopup1.png", cv2.IMREAD_UNCHANGED) # transparency
        self.CHECK_POPUP_BYTES = self.load_image("checkpopup.png")
        self.CALL_POPUP_BYTES = self.load_image("callpopup.png")
        self.BET_POPUP_BYTES = self.load_image("betpopup.png")
        self.RAISE_POPUP_BYTES = self.load_image("raisepopup.png")
        self.ALLIN_POPUP_BYTES = self.load_image("allinpopup.png")
        self.POST_POPUP_BYTES = self.load_image("postpopup.png")
        self.BIG_POPUP_BYTES = self.load_image("bigpopup.png")
        self.SMALL_POPUP_BYTES = self.load_image("smallpopup.png")

        self.POPUP_BYTES = [
            self.BASE_POPUP_BYTES,
            self.CHECK_POPUP_BYTES,
            self.CALL_POPUP_BYTES,
            self.BET_POPUP_BYTES,
            self.RAISE_POPUP_BYTES,
            self.ALLIN_POPUP_BYTES,
            self.POST_POPUP_BYTES,
            self.BIG_POPUP_BYTES,
            self.SMALL_POPUP_BYTES
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
            TJPopupTypes.SMALL: self.SMALL_POPUP_BYTES
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
                return pretty_str_to_int(text)
            except ValueError:
                return 0
            
    # due to triplejack having them all at the bottom, moved this out of generic impl for speedup.
    def __get_button_subsection(self, screenshot: cv2.typing.MatLike) -> tuple[int, int, int, int]:
        h = screenshot.shape[0]
        return (0, h - h // 5, screenshot.shape[1], h)

    def call_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.CALL_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def check_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.CHECK_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def bet_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.BET_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def fold_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.FOLD_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def raise_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(screenshot, self.RAISE_BUTTON_BYTES, threshold, self.__get_button_subsection(screenshot))
    
    def allin_button(self, screenshot: cv2.typing.MatLike, threshold=0.95) -> Union[tuple[int, int, int, int], None]:
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

    def set_seat_loc(self, loc: tuple[int, int]):
        self.seat_loc = (loc[0], int(loc[1] * 0.81))

    def stack_size(self, img: MatLike) -> int:
        if self.seat_loc is None:
            raise ValueError("seat_loc not set")
        left = self.seat_loc[0] + 56
        top = self.seat_loc[1] + 63
        right = left + 77
        bottom = top + 27
        number = self.ocr_text_from_image(img, (left, top, right, bottom), invert=True, brightness=0.5, contrast=3, erode=True)
        return pretty_str_to_int(number)

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

        for popup in self.POPUP_BYTES:
            for loc in self.template_detect(img, popup, threshold=0.95):
                ret.append(self.ident_near_popup(img, loc))
        
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
                return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
            else:
                allin_button = self.allin_button(img)
                if allin_button is not None:
                    loc = (allin_button[0] - 10, allin_button[3], allin_button[2] + 10, allin_button[3] + (allin_button[3] - allin_button[1]) + 3)
                    return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
                else:
                    raise RuntimeError("Not my turn or couldn't find current bet")


    def min_bet(self, img: MatLike) -> int:
        # when you call this, check if current_bet >= stack_size, if it is, save time by not calling this method
        bet_button = self.bet_button(img)
        if bet_button is not None:
            loc = (bet_button[0] - 10, bet_button[3], bet_button[2] + 10, bet_button[3] + (bet_button[3] - bet_button[1]) + 3)
            return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
        else:
            raise_button = self.raise_button(img)
            if raise_button is not None:
                loc = (raise_button[0] - 10, raise_button[3], raise_button[2] + 10,
                       raise_button[3] + (raise_button[3] - raise_button[1]) + 3)
                return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
            else:
                allin_button = self.allin_button(img)
                if allin_button is not None:
                    loc = (allin_button[0] - 10, allin_button[3], allin_button[2] + 10,
                           allin_button[3] + (allin_button[3] - allin_button[1]) + 3)
                    return pretty_str_to_int(self.ocr_text_from_image(img, loc, contrast=3))
                else:
                    raise RuntimeError("Not my turn or couldn't find current bet")

    def table_players(self, img: MatLike) -> list:
        modified_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # crank that bri-con!
        brightness = 120
        contrast = 120
        modified_img = np.int16(modified_img)
        modified_img = modified_img * (contrast / 127 + 1) - contrast + brightness
        modified_img = np.clip(modified_img, 0, 255)
        modified_img = np.uint8(modified_img)

        # find lines using houghlinesp
        gray = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)

        # show
        cv2.imshow("img", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        edges = cv2.Canny(gray, 100, 200, apertureSize=3)

        # show
        cv2.imshow("img", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        lines: list[tuple[int, int, int, int]] = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 90, threshold=50, minLineLength=200, maxLineGap=20)
    

        # filter out lines of length more than 230
        lines = list(filter(lambda x: math.sqrt((x[0][0] - x[0][2]) ** 2 + (x[0][1] - x[0][3]) ** 2) < 230, lines))
        # filter out lines whose y1 and y2 arent within 5 pixels
        lines = list(filter(lambda x: abs(x[0][1] - x[0][3]) < 5, lines))

        debug_image = modified_img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("img", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        players = []
        points = []
        # show lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            point1 = (x1, y1)
            if any([abs(x1 - x) < 20 and abs(y1 - y) < 20 for x, y in points]):
                continue
            point2 = (x2, y2)
            if any([abs(x2 - x) < 20 and abs(y2 - y) < 20 for x, y in points]):
                continue
            top = y1 - 20
            bottom = y1 + 20
            left = x1
            right = x2
            name = self.ocr_text_from_image(modified_img, (left, top, right, bottom), invert=True, brightness=0.3, contrast=3, allowed_chars=False, scale=50)
            players.append(name)
            points.append(point1)
            points.append(point2)
        return players

    def active_players(self, img: MatLike) -> list:
        # decrease the brightness of green pixels (the board)
        less_green_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(less_green_img, np.array([35, 100, 100]), np.array([70, 255, 255]))
        factor = 0.4
        less_green_img[..., 2] = less_green_img[..., 2] * (1 - green_mask / 255 * (1 - factor))
        # increase the brightness of non-green pixels
        less_green_img[..., 2] = less_green_img[..., 2] * (1 + green_mask / 255 * factor)

        less_green_img = cv2.cvtColor(less_green_img, cv2.COLOR_HSV2BGR)

        # blur to smooth out circles
        modified_img = cv2.GaussianBlur(less_green_img, (21, 21), 0)

        # crank that bri-con!
        brightness = 120
        contrast = 120
        modified_img = np.int16(modified_img)
        modified_img = modified_img * (contrast / 127 + 1) - contrast + brightness
        modified_img = np.clip(modified_img, 0, 255)
        modified_img = np.uint8(modified_img)

        # cv2.imshow("img", modified_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, binary_img = cv2.threshold(cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        height, width = binary_img.shape
        binary_img[3 * height // 10: int(5.5 * height // 10),
        width // 5: 4 * width // 5] = 0  # black out the community cards
        binary_img[4 * height // 5:, :] = 0  # black out the hole cards

        # show
        # cv2.imshow("img", binary_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 1, 140, param1=50, param2=10, minRadius=100, maxRadius=120)
        if circles is None:
            return []
        players = []
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            # print(circle)
            # show circle
            # cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # cv2.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            center = (circle[0], circle[1])
            top_y = int(center[1] + circle[2] * 0.65)
            bottom_y = int(top_y + circle[2] * (1/2.5))
            left_x = int(center[0] - circle[2])
            right_x = int(center[0] + circle[2])
            name = self.ocr_text_from_image(img, (left_x, top_y, right_x, bottom_y), invert=True, brightness=0.2, contrast=5, allowed_chars=False, scale=50)
            # print(name)
            players.append(name)

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
        

    def get_full_cards(self, img: MatLike, hole=False, subsection: Union[tuple[int, int, int, int], None]=None) -> list[tuple[Card, tuple[int, int, int, int]]]:
        if hole:
            suits = self.find_hole_suits(img, subsection=subsection)
        else:
            suits = self.find_community_suits(img, subsection=subsection)

        ret = []

        for key, locs in suits.items():
            for loc in locs:
                w, h = loc[2] - loc[0], loc[3] - loc[1]

                text_area = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3] - h,
                ) 

              
                text = self.ocr_text_from_image(img, text_area, psm=7, contrast=1.5)
      
                loc = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3],
                )

                full_card_str = f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}"

                img1 = cv2.rectangle(img.copy(), (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)
                cv2.putText(img1, full_card_str, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.imshow("img", img1)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if text == "":
                    cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                    cv2.imwrite("error.png", img)
                    raise ValueError("OCR failed to find card's text")

                try:
                    ret.append((Card.new(full_card_str), loc))
                except KeyError as e:
                    cv2.putText(img, full_card_str, (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                    cv2.imwrite("error.png", img)
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
        subsection = (w // 4, h // 4, w // 4 * 3, h // 4 * 3)
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
    import os
    import time

    folder = "triplejack/new/base/tests"
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
          
        
    

       

