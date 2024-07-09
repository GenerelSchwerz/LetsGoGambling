# lazy code for now


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
                community_clubs=("club.png", True),
                community_spades=("spade.png", True),
                hole_hearts=("hole_heart.png", True),
                hole_diamonds=("hole_diamond2.png", True),
                hole_clubs=("hole_club2.png", True),
                hole_spades=("hole_spade1.png", True),
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

    def load_images(self):
        super().load_images()

        self.POT_BYTES = self.load_image("pot.png")
        self.MAIN_POT_BYTES = self.load_image("mainpot.png")
        self.SIDE_POT_BYTES = self.load_image("sidepot.png")

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
        """
        TODO handle multiple pots
        """

        def ident_near(img, loc: tuple[int, int, int, int]):
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
                return ident_near(img, main_pot)

            sum = 0
            for pot in side_pots:
                sum += ident_near(img, pot)
                
            return sum + ident_near(img, main_pot)
        
        else:
            return ident_near(img, single_pot)


    def total_pot(self, img: MatLike) -> int:
        return super().total_pot()
    

    # the image you talked abt, hole card, a 9 is being detected as a 10, my bad (I think its being a zero)
    # can you commit and push so i can get my hole cards working
    def current_bet(self, img: MatLike) -> int:
        return super().current_bet()

    def min_bet(self, img: MatLike) -> int:
        return super().min_bet()

    def table_players(self, img: MatLike) -> list:
        return super().table_players()

    def active_players(self, img: MatLike) -> list:
        return super().active_players()

    def big_blinds(self, img: MatLike) -> int:
        return super().big_blinds()

    def get_full_cards(self, img: MatLike, loc: tuple, hole=False) -> list[tuple[Card, tuple[int, int, int, int]]]:
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
                # wut im testing on test-1720483487.png (with new images)

                # the new images fixed everything lol

                text = self.ocr_text_from_image(img, subsection, psm=10, brightness=4, contrast=2)
                print(f"found: {card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}")

                loc = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3],
                )

                if text == "":
                    cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                    cv2.imwrite("error.png", img)
                    raise ValueError("OCR failed to find card's text")

                ret.append((Card.new(f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}"), loc))

        # sort ret by x position (want left to right)
        ret.sort(key=lambda x: x[1][0])

        return ret

    def community_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the middle 4th of the screen

        h = img.shape[0]
        img = img[img.shape[0] // 4 : img.shape[0] // 4 * 3, :, :]

        ret = self.get_full_cards(img, None)

        # shift the y position of the cards up 1/2th of the height of the image
        return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4, x[1][2], x[1][3] + h // 4)), ret))



    def community_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.community_cards_and_locs(img)))


    def hole_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int, int, int]]]:
        # resize image to the bottom 4th of the screen
        h = img.shape[0]
        img = img[img.shape[0] // 4 * 3 :, :, :]
        ret = self.get_full_cards(img, None, hole=True)

        # shift the y position of the cards up 1/2th of the height of the image
        return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4 * 3, x[1][2], x[1][3] + h // 4 * 3)), ret))

    def hole_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.hole_cards_and_locs(img)))
    
    def sit_button(self, img: MatLike) -> list[tuple[int, int, int, int]]:
        return self.find_sit_button(img)
    

    
    def call_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.CALL_BUTTON_BYTES)

    def raise_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.RAISE_BUTTON_BYTES)
    
    def check_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.CHECK_BUTTON_BYTES)
    
    def bet_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.BET_BUTTON_BYTES)
    
    def fold_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.FOLD_BUTTON_BYTES)
    
    def allin_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        return self.ident_one_template(img, self.ALLIN_BUTTON_BYTES)
    


       


if __name__ == "__main__":

    detect = TJPokerDetect()
    detect.load_images()

    img = cv2.imread("triplejack/new/base/tests/bad.png", cv2.IMREAD_COLOR)


    info = detect.community_cards_and_locs(img)

    for card, loc in info:
        Card.print_pretty_card(card)
        cv2.putText(img, Card.int_to_str(card), (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)


    info1 = detect.hole_cards_and_locs(img)
    
    for card, loc in info1:
        Card.print_pretty_card(card)
        cv2.putText(img, Card.int_to_str(card), (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    sit_locs = detect.find_sit_button(img)
    print(sit_locs)

    for loc in sit_locs:

        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    call_loc = detect.call_button(img)
    if call_loc is not None:
        print("call button found")
        cv2.rectangle(img, (call_loc[0], call_loc[1]), (call_loc[2], call_loc[3]), (0, 255, 0), 2)
    else:
        print("no call button found")


    check_loc = detect.check_button(img)
    if check_loc is not None:
        print("check button found")
        cv2.rectangle(img, (check_loc[0], check_loc[1]), (check_loc[2], check_loc[3]), (0, 255, 0), 2)
    else:
        print("no check button found")

    bet_loc = detect.bet_button(img)
    if bet_loc is not None:
        print("bet button found")
        cv2.rectangle(img, (bet_loc[0], bet_loc[1]), (bet_loc[2], bet_loc[3]), (0, 255, 0), 2)
    else:
        print("no bet button found")

    fold_loc = detect.fold_button(img)
    if fold_loc is not None:
        print("fold button found")
        cv2.rectangle(img, (fold_loc[0], fold_loc[1]), (fold_loc[2], fold_loc[3]), (0, 255, 0), 2)
    else:
        print("no fold button found")

    raise_loc = detect.raise_button(img)
    if raise_loc is not None:
        cv2.rectangle(img, (raise_loc[0], raise_loc[1]), (raise_loc[2], raise_loc[3]), (0, 255, 0), 2)
    else:
        print("no raise button found")

    allin_loc = detect.allin_button(img)
    if allin_loc is not None:
        print("allin button found")
        cv2.rectangle(img, (allin_loc[0], allin_loc[1]), (allin_loc[2], allin_loc[3]), (0, 255, 0), 2)
    else:
        print("no allin button found")

    
    pot = detect.middle_pot(img)
    print(pot)


    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       

