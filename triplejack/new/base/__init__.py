# lazy code for now


from typing import Union
from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike

from treys import Card

from .utils import *


class TJPokerDetect(PokerImgDetect, PokerDetection):

    def __init__(self) -> None:
        super().__init__(
            opts=PokerImgOpts(
                folder_path="triplejack/new/base/imgs",
                sit_button="sit.png",
                community_hearts="heart.png",
                community_diamonds="diamond.png",
                community_clubs="club.png",
                community_spades="spade.png",
                hole_hearts="hole_heart.png",
                hole_diamonds="hole_diamond.png",
                hole_clubs="hole_club.png",
                hole_spades="hole_spade.png",
            )
        )

        self.CALL_BUTTON_BYTES = None

    def load_images(self):
        super().load_images()

        self.CALL_BUTTON_BYTES = self.load_image("callbutton.png")

    def stack_size(self, img: MatLike) -> int:
        return super().stack_size()

    def middle_pot(self, img: MatLike) -> int:
        return super().middle_pot()

    def total_pot(self, img: MatLike) -> int:
        return super().total_pot()

    def current_bet(self, img: MatLike) -> int:
        return super().current_bet()

    def min_bet(self, img: MatLike) -> int:
        return super().min_bet()

    def table_players(self, img: MatLike) -> list:
        return super().table_players()

    def hole_cards(self, img: MatLike) -> list:
        return super().hole_cards()

    def active_players(self, img: MatLike) -> list:
        return super().active_players()

    def big_blinds(self, img: MatLike) -> int:
        return super().big_blinds()

    def get_full_cards(self, img: MatLike, loc: tuple) -> list[Card]:
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

                text = self.ocr_text_from_image(img, subsection)
                print(f"found: {card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}")

                # merge the original loc and subsection together for a larger subsection
                loc = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3],
                )

                if text == "":
                    cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
                    cv2.imwrite("error.png", img)
                    raise ValueError("Could not find card's text!")
                
                ret.append((Card.new(f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}"), loc))

        # sort ret by x position (want left to right)
        ret.sort(key=lambda x: x[1][0])

        return ret


    
    def community_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int]]]:
        # resize image to the middle 4th of the screen
        h = img.shape[0]
        img = img[img.shape[0] // 4 : img.shape[0] // 4 * 3, :, :]
        ret = self.get_full_cards(img, None)

        # shift 1/4th down
        return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4, x[1][2], x[1][3] + h // 4)), ret))

    def community_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.community_cards_and_locs(img)))


    def hole_cards_and_locs(self, img: MatLike) -> list[tuple[Card, tuple[int, int]]]:
        # resize image to the bottom 4th of the screen
        h = img.shape[0]
        img = img[img.shape[0] // 4 * 3 :, :, :]
        ret = self.get_full_cards(img, None)

        # shift 3/4ths down
        return list(map(lambda x: (x[0], (x[1][0], x[1][1] + h // 4 * 3, x[1][2], x[1][3] + h // 4 * 3)), ret))
    
    def hole_cards(self, img: MatLike) -> list[Card]:
        return list(map(lambda x: x[0], self.hole_cards_and_locs(img)))
    
    def sit_button(self, img: MatLike) -> list[tuple[int, int, int, int]]:
        return self.find_sit_button(img)
    
    def call_button(self, img: MatLike) -> Union[tuple[int, int, int, int], None]:
        locs = self.template_detect(img, self.CALL_BUTTON_BYTES, threshold=0.9)

        if len(locs) == 0:
            return None
        
        return locs[0]



if __name__ == "__main__":

    detect = TJPokerDetect()
    detect.load_images()

    img = cv2.imread("triplejack/new/base/tests/download (1).png", cv2.IMREAD_COLOR)
    for card, loc in detect.community_cards_and_locs(img):
        Card.print_pretty_card(card)
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    for card, loc in detect.hole_cards_and_locs(img):
        Card.print_pretty_card(card)
        print(loc)
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    sit_locs = detect.find_sit_button(img)
    print(sit_locs)

    for loc in sit_locs:
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 2)

    call_loc = detect.call_button(img)
    if call_loc is not None:
        cv2.rectangle(img, (call_loc[0], call_loc[1]), (call_loc[2], call_loc[3]), (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       

