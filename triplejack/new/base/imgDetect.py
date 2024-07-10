# lazy code for now


import math
from typing import Union
from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike
import cv2

from treys import Card

from .utils import *


class TJPokerDetect(PokerImgDetect, PokerDetection):

    def __init__(self) -> None:
        super().__init__(
            opts=PokerImgOpts(
                folder_path="triplejack/new/base/imgs",
                sit_button=("sit.png", False),
                community_hearts=("heart.png", True),
                community_diamonds=("diamond1.png", True),
                community_clubs=("club1.png", True),
                community_spades=("spade.png", True),
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

    def __ident_near_popup(self, img, loc: tuple[int, int, int, int]):
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

    def find_community_suits(self, ss1: cv2.typing.MatLike, threshold=0.77) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return super().find_community_suits(ss2, threshold)

    def find_hole_suits(self, ss1: cv2.typing.MatLike, threshold=0.77) -> dict[str, list[tuple[int, int, int, int]]]:
        ss2 = cv2.cvtColor(ss1, cv2.COLOR_RGB2GRAY)
        _, ss2 = cv2.threshold(ss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return super().find_hole_suits(ss2, threshold)


    def stack_size(self, img: MatLike) -> int:
        return super().stack_size()

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

        single_pot = self.ident_one_template(img, self.POT_BYTES)

        if single_pot is None:

            main_pot = self.ident_one_template(img, self.MAIN_POT_BYTES)

            # no pots currently visible
            if main_pot is None:
                return 0
            
            side_pots = self.template_detect(img, self.SIDE_POT_BYTES)
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
        return [self.__ident_near_popup(img, locs) for locs in self.template_detect(img, self.BASE_POPUP_BYTES, threshold=0.95)]
        
    def current_bet(self, img: MatLike) -> int:
        return max(self.current_bets(img))

    def min_bet(self, img: MatLike) -> int:
        return super().min_bet()

    def table_players(self, img: MatLike) -> list:
        return super().table_players()

    def active_players(self, img: MatLike) -> list:
        return super().active_players()

    def big_blind(self, img: MatLike) -> int:
        big_blind_popup = self.ident_one_template(img, self.BIG_POPUP_BYTES)
        if big_blind_popup is None:
            return -1
        
        return self.__ident_near_popup(img, big_blind_popup)
    
    def small_blind(self, img: MatLike) -> int:
        small_blind_popup = self.ident_one_template(img, self.SMALL_POPUP_BYTES)
        if small_blind_popup is None:
            return -1
        
        return self.__ident_near_popup(img, small_blind_popup)
        

    def get_full_cards(self, img: MatLike, hole=False) -> list[tuple[Card, tuple[int, int, int, int]]]:
        if hole:
            suits = self.find_hole_suits(img)
        else:
            suits = self.find_community_suits(img)

        ret = []

        for key, locs in suits.items():
            for loc in locs:
                w, h = loc[2] - loc[0], loc[3] - loc[1]

                subsection = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3] - h,
                ) 

              
                text = self.ocr_text_from_image(img, subsection, psm=7, contrast=1.5)
      
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

    def community_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the middle 4th of the screen

        h = img.shape[0]
        w = img.shape[1]
        img = img[h // 4 : h // 4 * 2, w // 4: w // 4 * 3, :]

        ret = self.get_full_cards(img)

        # shift the y position of the cards up 1/2th of the height of the image
        return list(map(lambda x: (x[0], (x[1][0] + w // 4, x[1][1] + h // 4, x[1][2] + w // 4, x[1][3] + h // 4)), ret))



    def community_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.community_cards_and_locs(img)))


    def hole_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the bottom 4th of the screen
        h = img.shape[0]
        w = img.shape[1]
        img = img[img.shape[0] // 4 * 3 :, : w // 4, :]
        ret = self.get_full_cards(img, hole=True)

    
        # shift the y position of the cards up 1/2th of the height of the image
        return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4 * 3, x[1][2], x[1][3] + h // 4 * 3)), ret))

    def hole_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.hole_cards_and_locs(img)))
    




       
def report_info(detector: TJPokerDetect, ss: str | cv2.typing.MatLike):

    if isinstance(ss, str):
        img = cv2.imread(ss, cv2.IMREAD_COLOR)
       
    else:
         img = ss
       
    img2 = img.copy()
    # i'm just gonna go pass out i think please call me later if ur still up
    # i'm probably going to call it an early night
    # if you need me for anything let me know
    # and if you have any tasks you want to relegate to me that are either really quick and simple for tonight or other tasks for tmrw, lmk
    # love you <3
    # oh btw if you end up super night owling it and get to the point where u wanna test a decisionmaker,
    # check the latest commit in my repository, the method signature looks like 
    # make_decision([holecard1, holecard2], board_cards, stack_size, pot_value, current_bet,
                                        #  min_bet, num_opponents, self.big_blind, middle_pot_value, self.flags)
# and you can make an abstract class or some shit idk and replace the "returns" inside PokerDecisionMaker with returning an enum decision thing or something anyway bye love you


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
    current_bet = 0
    mid_pot = detector.middle_pot(img)
    print(mid_pot)

    tot_pot = detector.total_pot(img)
    print(tot_pot)

    current_bet = detector.current_bets(img)
    print(current_bet)

    filename = ss if isinstance(ss, str) else "current image"

    

    cv2.imshow(f"{filename} ({img.shape[0]}x{img.shape[1]}) | Stage: {PokerStages.to_str(cards_to_stage(info))} | mid pot: {mid_pot} | tot pot: {tot_pot} | current bet: {current_bet} | Took: {int((time.time() - now) * 1000) / 1000} sec", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect = TJPokerDetect()
    detect.load_images()
    import os
    import time

    # for all files in a directory, run the report_info function
    files = os.listdir("triplejack/new/base/tests")
    # files = [
    #     "test-1720544837.png"
    # ]
    files = sorted(files)
    # files.reverse()
    print(files)
    for filename in files:
        if filename.endswith(".png"):
            path =  os.path.join("triplejack/new/base/tests", filename)
            print("running report_info on",path)

           
            report_info(detect, path)
          
        
    

       

