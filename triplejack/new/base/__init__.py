# lazy code for now


from ...abstract import PokerDetection
from ...abstract.impl import *
from cv2.typing import MatLike

from treys import Card

from .utils import *


class TJPokerDetect(PokerImgDetect, PokerDetection):

    def __init__(self) -> None:
        super().__init__(
            opts=PokerImgOpts(
                folder_path="triplejack/new/base/tests",
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

    def community_cards(self, img: MatLike) -> list:

        # resize image to the middle 4th of the screen

        img = img[img.shape[0] // 4 : img.shape[0] // 4 * 3, :, :]

        suits = self.find_community_suits(img)

        ret = {}

        for key, locs in suits.items():
            for loc in locs:
                w, h = loc[2] - loc[0], loc[3] - loc[1]

                cv2.imshow("img", img[loc[1] : loc[3], loc[0] : loc[2]])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                subsection = (
                    loc[0] - w // 6,
                    loc[1] - h - h // 6,
                    loc[2] + w // 6,
                    loc[3] - h,
                )

                text = self.ocr_text_from_image(img, subsection)

                if ret.get(key) is None:
                    ret[key] = []
                ret[key].append(
                    Card.new(f"{card_to_abbrev(text)}{suit_full_name_to_abbrev(key)}")
                )

        return ret


if __name__ == "__main__":

    detect = TJPokerDetect()
    detect.load_images()

    img = cv2.imread("triplejack/new/base/tests/download (1).png", cv2.IMREAD_COLOR)
    for suit, cards in detect.community_cards(img).items():
        Card.print_pretty_cards(cards)
            

