# lazy code for now
from typing import Union
from .utils import *

import cv2
import numpy as np

from PIL import Image, ImageEnhance
from PIL.Image import Resampling

import pytesseract


class PokerImgDetect:

    def __init__(self, opts: PokerImgOpts) -> None:
        self.opts = opts
 
        self.SIT_BUTTON_BYTES = None
    
        self.COMMUNITY_HEART_SUIT_BYTES = None
        self.COMMUNITY_DIAMONDS_SUIT_BYTES = None
        self.COMMUNITY_CLUBS_SUIT_BYTES = None
        self.COMMUNITY_SPADES_SUIT_BYTES = None

        self.HOLE_HEART_SUIT_BYTES = None
        self.HOLE_DIAMONDS_SUIT_BYTES = None
        self.HOLE_CLUBS_SUIT_BYTES = None
        self.HOLE_SPADES_SUIT_BYTES = None


    @staticmethod
    def template_detect(fullimg: cv2.typing.MatLike, wanted: cv2.typing.MatLike, threshold=0.77):
        # check if fullimg is color or grayscale
        if len(fullimg.shape) == 2:
            w, h = wanted.shape[::]
        elif len(fullimg.shape) == 3:
            w, h = wanted.shape[:-1]
        else:
            raise ValueError("Invalid image shape")

        res = cv2.matchTemplate(fullimg, wanted, cv2.TM_CCOEFF_NORMED)
 
        loc = np.where(res >= threshold)

        zipped = np.array(
            [[pt[0], pt[1], pt[0] + h, pt[1] + w] for pt in zip(*loc[::-1])]
        )

        return non_max_suppression_slow(zipped, 0.3) 

    @staticmethod
    def ident_one_template(img: cv2.typing.MatLike, wanted: cv2.typing.MatLike, threshold=0.9) -> Union[tuple[int, int, int, int], None]:
        locs = PokerImgDetect.template_detect(img, wanted, threshold=threshold)

        if len(locs) == 0:
            return None
        
        return locs[0]

    def load_image(self, name: str, flags = cv2.IMREAD_COLOR):
        return cv2.imread(f"{self.opts.folder_path}/{name}", flags=flags)

    def load_images(self):

        self.SIT_BUTTON_BYTES = self.load_image(self.opts.sit_button)

        self.COMMUNITY_HEART_SUIT_BYTES = self.load_image(self.opts.community_hearts)
        self.COMMUNITY_DIAMONDS_SUIT_BYTES = self.load_image(self.opts.community_diamonds)
        self.COMMUNITY_CLUBS_SUIT_BYTES = self.load_image(self.opts.community_clubs)
        self.COMMUNITY_SPADES_SUIT_BYTES = self.load_image(self.opts.community_spades)

        self.HOLE_HEART_SUIT_BYTES = self.load_image(self.opts.hole_hearts)
        self.HOLE_DIAMONDS_SUIT_BYTES = self.load_image(self.opts.hole_diamonds)
        self.HOLE_CLUBS_SUIT_BYTES = self.load_image(self.opts.hole_clubs)
        self.HOLE_SPADES_SUIT_BYTES = self.load_image(self.opts.hole_spades)


    def find_sit_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> list[tuple[int, int, int, int]]:
        return self.template_detect(screenshot, self.SIT_BUTTON_BYTES, threshold=threshold)

    def find_community_suits(self, ss1: cv2.typing.MatLike, threshold=0.77) -> dict[str, list[tuple[int, int, int, int]]]:
        hearts = self.template_detect(ss1, self.COMMUNITY_HEART_SUIT_BYTES, threshold=threshold)
        diamonds = self.template_detect(ss1, self.COMMUNITY_DIAMONDS_SUIT_BYTES, threshold=threshold)
        clubs = self.template_detect(ss1, self.COMMUNITY_CLUBS_SUIT_BYTES, threshold=threshold)
        spades = self.template_detect(ss1, self.COMMUNITY_SPADES_SUIT_BYTES, threshold=threshold)
        return {
            "hearts": hearts,
            "diamonds": diamonds,
            "clubs": clubs,
            "spades": spades
        }
        
    def find_hole_suits(self, screenshot: cv2.typing.MatLike, threshold=0.85) -> dict[str, list[tuple[int, int, int, int]]]:
        hearts = self.template_detect(screenshot, self.HOLE_HEART_SUIT_BYTES, threshold=threshold)
        diamonds = self.template_detect(screenshot, self.HOLE_DIAMONDS_SUIT_BYTES, threshold=threshold)
        clubs = self.template_detect(screenshot, self.HOLE_CLUBS_SUIT_BYTES, threshold=threshold)
        spades = self.template_detect(screenshot, self.HOLE_SPADES_SUIT_BYTES, threshold=threshold)
        return {
            "hearts": hearts,
            "diamonds": diamonds,
            "clubs": clubs,
            "spades": spades
        }
    
# ===================
# OCR nonsense
# ===================

    def erase_edges(self, binary_image: cv2.typing.MatLike):
        old_image = binary_image.copy()
        target_value = 0
        replacement_value = 255

        # not recursive bc stack overflow
        def flood_fill(binary_image, i, j):
            stack = [(i, j)]
            while stack:
                i, j = stack.pop()
                if i < 0 or i >= binary_image.shape[0] or j < 0 or j >= binary_image.shape[1]:
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
            print('Erased edges flood fill failed, whole image is white')
            return old_image
        else:
            return binary_image

    def eliminate_isolated_pixels(self, image):
        output_image = image.copy()

        rows, cols = image.shape

        # Define the 8 neighbor directions (N, NE, E, SE, S, SW, W, NW)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        # Iterate over each pixel in the image
        for r in range(rows):
            for c in range(cols):
                # Only consider black pixels (value 0)
                if image[r, c] == 0:
                    white_count = 0
                    # Count white neighbors
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        # Check if the neighbor is within the bounds of the image
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if image[nr, nc] == 255:
                                white_count += 1
                    # If a black pixel is surrounded by 5 or more white pixels, convert it to white
                    if white_count >= 5:
                        output_image[r, c] = 255
        return output_image



    def ocr_text_from_image(self, screenshot: np.ndarray, location: tuple[int, int, int, int], rotation_angle=0, psm=7, invert=False, erode=False, brightness=0.0, contrast=0.0):

        image = screenshot[location[1]:location[3], location[0]:location[2]]

        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if rotation_angle != 0:
            image = Image.fromarray(image)
            image = image.rotate(rotation_angle, resample=Resampling.BICUBIC, fillcolor=(255, 255, 255))
            image = np.array(image)

        if brightness > 0.0:
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

        if contrast > 0:
            img = Image.fromarray(np.uint8(image))
            contrasted_img = ImageEnhance.Contrast(img).enhance(contrast)
            image = np.array(contrasted_img)


        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


        # scale height to 33 pixels using bicubic resampling
        gray = cv2.resize(gray, (0, 0), fx=40 / gray.shape[0], fy=40 / gray.shape[0], interpolation=cv2.INTER_CUBIC)

        # i know this effect better as feathering but everyone calls it erosion
        if erode:
            gray = cv2.bitwise_not(gray)
            gray = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.erode(gray, kernel, iterations=1)
            gray = cv2.resize(gray, (0, 0), fx=1 / 2.0, fy=1 / 2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.bitwise_not(gray)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if invert:
            binary = cv2.bitwise_not(binary)

        binary = self.erase_edges(binary)
        binary = self.eliminate_isolated_pixels(binary)
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        result = pytesseract.image_to_string(binary, lang='eng', config=CUSTOM_CONFIG.format(psm=psm)).strip()
   
        # can you tell what number has given me hours of trouble with tesseract and TJ font
        if any(result == char for char in ['0', 'O', '1', 'I', "170", "I70", "17O", "I7O", "70", "1O", "IO", "I0", "7O"]):
            return '10'
        return result


    

if __name__ == "__main__":
    poker_img_detect = PokerImgDetect("tests/")
    poker_img_detect.load_images()

    org_img = open("tests/full.png", "rb").read()
    raw = np.frombuffer(org_img, np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)

    img1, img = cv2.threshold(img, 127, 255, 0)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    locations = poker_img_detect.detect_sit_button(img, threshold=0.77)

    for pt in locations:
        print("sit", pt)
        cv2.rectangle(img2, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255), 2)

    locations2 = poker_img_detect.find_community_suits(img, threshold=0.77)

    for key, locations in locations2.items():
        for pt in locations:
            print("community", key, pt)
            cv2.rectangle(img2, (pt[0], pt[1]), (pt[2], pt[3]), (0, 255, 0), 2)
            poker_img_detect.card_number(img, pt)

    locations3 = poker_img_detect.find_hole_suits(img, threshold=0.85)

    for key, locations in locations3.items():
        for pt in locations:
            print("hole", key, pt)
            cv2.rectangle(img2, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 2)

            poker_img_detect.card_number(img2, pt)


    cv2.imwrite("result.png", img2)
    cv2.imwrite("used.png", img)
    pass
