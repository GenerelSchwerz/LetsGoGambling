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

        self.CHECK_BUTTON_BYTES = None
        self.CALL_BUTTON_BYTES = None
        self.BET_BUTTON_BYTES = None
        self.FOLD_BUTTON_BYTES = None
        self.RAISE_BUTTON_BYTES = None
        self.ALLIN_BUTTON_BYTES = None

        self.__initalized = False

    def is_initialized(self):
        return self.__initalized    

    @staticmethod
    def template_detect(fullimg: cv2.typing.MatLike, wanted: cv2.typing.MatLike, threshold=0.77, subsection: Union[tuple[int,int,int,int], None] = None) -> list[tuple[int, int, int, int]]:
        
        if subsection is not None:
            fullimg = fullimg[subsection[1]:subsection[3], subsection[0]:subsection[2]]
            # cv2.imshow("img", fullimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        w = wanted.shape[0]
        h = wanted.shape[1]

        # automatically handle transparency 
        if len(wanted.shape) == 3 and wanted.shape[2] == 4:
            # assuming wanted is currently in BGRA format, convert back to BGR (color)
            base = wanted[:,:,0:3]
            alpha = wanted[:,:,3]
            mask = cv2.merge([alpha, alpha, alpha])
            res = cv2.matchTemplate(fullimg, base, cv2.TM_CCOEFF_NORMED, mask=mask)
        else:
            # check if template is larger than image
            if w > fullimg.shape[0] or h > fullimg.shape[1]:
                print(subsection)
                print(fullimg.shape, wanted.shape)
                cv2.imshow("img", fullimg)
                cv2.imshow("wanted", wanted)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            res = cv2.matchTemplate(fullimg, wanted, cv2.TM_CCOEFF_NORMED)

        loc = np.where((res >= threshold) &  np.isfinite(res))
        
        if len(loc[0]) == 0:
            return []
        

        if subsection is not None:
            zipped = np.array(
                [[pt[0] + subsection[0], pt[1] + subsection[1], pt[0] + h + subsection[0], pt[1] + w + subsection[1]] for pt in zip(*loc[::-1])]
            )

        else:

            zipped = np.array(
                [[pt[0], pt[1], pt[0] + h, pt[1] + w] for pt in zip(*loc[::-1])]
            )

        return non_max_suppression_slow(zipped, 0.01) 

    @staticmethod
    def ident_one_template(img: cv2.typing.MatLike, wanted: cv2.typing.MatLike, threshold=0.9, subsection: Union[tuple[int,int,int,int], None]=None) -> Union[tuple[int, int, int, int], None]:
        locs = PokerImgDetect.template_detect(img, wanted, threshold=threshold, subsection=subsection)

        if len(locs) == 0:
            return None
        
        return locs[0]

    def load_image(self, name: str, flags = cv2.IMREAD_COLOR, binary = False):
        image = cv2.imread(f"{self.opts.folder_path}/{name}", flags=flags if not binary else cv2.IMREAD_GRAYSCALE)
        if binary:
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

    def load_images(self):

        self.SIT_BUTTON_BYTES = self.load_image(self.opts.sit_button[0], binary=self.opts.sit_button[1])

        self.COMMUNITY_HEART_SUIT_BYTES = self.load_image(self.opts.community_hearts[0], binary=self.opts.community_hearts[1])
        self.COMMUNITY_DIAMONDS_SUIT_BYTES = self.load_image(self.opts.community_diamonds[0], binary=self.opts.community_diamonds[1])
        self.COMMUNITY_CLUBS_SUIT_BYTES = self.load_image(self.opts.community_clubs[0], binary=self.opts.community_clubs[1])
        self.COMMUNITY_SPADES_SUIT_BYTES = self.load_image(self.opts.community_spades[0], binary=self.opts.community_spades[1])

        self.HOLE_HEART_SUIT_BYTES = self.load_image(self.opts.hole_hearts[0], binary=self.opts.hole_hearts[1])
        self.HOLE_DIAMONDS_SUIT_BYTES = self.load_image(self.opts.hole_diamonds[0], binary=self.opts.hole_diamonds[1])
        self.HOLE_CLUBS_SUIT_BYTES = self.load_image(self.opts.hole_clubs[0], binary=self.opts.hole_clubs[1])
        self.HOLE_SPADES_SUIT_BYTES = self.load_image(self.opts.hole_spades[0], binary=self.opts.hole_spades[1])

        self.CHECK_BUTTON_BYTES = self.load_image(self.opts.check_button[0], binary=self.opts.check_button[1])
        self.CALL_BUTTON_BYTES = self.load_image(self.opts.call_button[0], binary=self.opts.call_button[1])
        self.BET_BUTTON_BYTES = self.load_image(self.opts.bet_button[0], binary=self.opts.bet_button[1])
        self.FOLD_BUTTON_BYTES = self.load_image(self.opts.fold_button[0], binary=self.opts.fold_button[1])
        self.RAISE_BUTTON_BYTES = self.load_image(self.opts.raise_button[0], binary=self.opts.raise_button[1])
        self.ALLIN_BUTTON_BYTES = self.load_image(self.opts.allin_button[0], binary=self.opts.allin_button[1])

        self.__initalized = True


    def sit_buttons(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> list[tuple[int, int, int, int]]:
        return self.template_detect(screenshot, self.SIT_BUTTON_BYTES, threshold)

    def call_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.CALL_BUTTON_BYTES, threshold)
    
    def check_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.CHECK_BUTTON_BYTES, threshold)
    
    def bet_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.BET_BUTTON_BYTES, threshold)
    
    def fold_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.FOLD_BUTTON_BYTES, threshold)
    
    def raise_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.RAISE_BUTTON_BYTES, threshold)
    
    def allin_button(self, screenshot: cv2.typing.MatLike, threshold=0.77) -> tuple[int, int, int, int]:
        return self.ident_one_template(screenshot, self.ALLIN_BUTTON_BYTES, threshold)


    def find_community_suits(self, ss1: cv2.typing.MatLike, threshold=0.77, subsection: Union[tuple[int, int, int, int], None]=None) -> dict[str, list[tuple[int, int, int, int]]]:
        hearts = self.template_detect(ss1, self.COMMUNITY_HEART_SUIT_BYTES, threshold=threshold, subsection=subsection)
        diamonds = self.template_detect(ss1, self.COMMUNITY_DIAMONDS_SUIT_BYTES, threshold=threshold, subsection=subsection)    
        clubs = self.template_detect(ss1, self.COMMUNITY_CLUBS_SUIT_BYTES, threshold=threshold, subsection=subsection)
        spades = self.template_detect(ss1, self.COMMUNITY_SPADES_SUIT_BYTES, threshold=threshold, subsection=subsection)
        return {
            "hearts": hearts,
            "diamonds": diamonds,
            "clubs": clubs,
            "spades": spades
        }
        
    def find_hole_suits(self, screenshot: cv2.typing.MatLike, threshold=0.85, subsection: Union[tuple[int, int, int, int], None]=None) -> dict[str, list[tuple[int, int, int, int]]]:

        # cv2.imshow("img", screenshot)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        hearts = self.template_detect(screenshot, self.HOLE_HEART_SUIT_BYTES, threshold=threshold, subsection=subsection)
        diamonds = self.template_detect(screenshot, self.HOLE_DIAMONDS_SUIT_BYTES, threshold=threshold, subsection=subsection)
        clubs = self.template_detect(screenshot, self.HOLE_CLUBS_SUIT_BYTES, threshold=threshold, subsection=subsection)
        spades = self.template_detect(screenshot, self.HOLE_SPADES_SUIT_BYTES, threshold=threshold, subsection=subsection)
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
            # print('Erased edges flood fill failed, whole image is white')
            return old_image
        else:
            return binary_image

    def eliminate_isolated_pixels(self, image):
        output_image = image.copy()

        rows, cols = image.shape

        # Define the 8 neighbor directions (N, NE, E, SE, S, SW, W, NW)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        to_white_out = []
        # Iterate over each pixel in the image
        for r in range(rows):
            for c in range(cols):
                # Only consider black pixels (value 0)
                if output_image[r, c] == 0:
                    white_count = 0

                    # N, E, S, W
                    dirs = [False, False, False, False]

                    # Count white neighbors
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        # Check if the neighbor is within the bounds of the image
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if output_image[nr, nc] == 255:
                                white_count += 1
                                if dr == -1 and dc == 0:
                                    dirs[0] = True
                                elif dr == 1 and dc == 0:
                                    dirs[2] = True
                                if dc == 1 and dr == 0:
                                    dirs[1] = True
                                elif dc == -1 and dr == 0:
                                    dirs[3] = True
                    if white_count >= 5 and sum(dirs) > 2:
                        to_white_out.append((r, c))
        for r, c in to_white_out:
            output_image[r, c] = 255

        def count_black_in_line(pixels) -> int:
            return len([pixel for pixel in pixels if pixel == 0])

        def count_black_neighbours(binary_image, r, c, radius) -> int:
            count = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i == 0 and j == 0:
                        continue
                    if 0 <= c + i < binary_image.shape[1] and 0 <= r + j < binary_image.shape[0]:
                        if binary_image[r + j, c + i] == 0:
                            count += 1
            return count

        def is_isolated(up, down, left, right, threshold) -> bool:
            gamma_rays = [count_black_in_line(up), count_black_in_line(down),
                          count_black_in_line(left), count_black_in_line(right)]
            return sum(gamma_rays) <= threshold

        done = False
        while not done:
            done = True
            for c in range(cols):
                for r in range(rows):
                    # if the pixel is black
                    if output_image[r, c] == 0:
                        # get the lines in each direction not including x,y
                        up = output_image[max(0, r - 15):r, c] if r > 0 else []
                        down = output_image[r + 1:min(rows, r + 15), c] if r < rows - 1 else []
                        left = output_image[r, max(0, c - 15):c] if c > 0 else []
                        right = output_image[r, c + 1:min(cols, c + 15)] if c < cols - 1 else []
                        # if there's at most one black pixel in the way to the edge
                        black_neighbours_r2 = count_black_neighbours(output_image, r, c, 2)
                        black_neighbours_r4 = count_black_neighbours(output_image, r, c, 4)
                        if ((is_isolated(up, down, left, right, 4) and black_neighbours_r2 < 6)
                                or (is_isolated(up, down, left, right, 10) and black_neighbours_r4 < 9)):
                            # print('Isolated pixel found at ', c, r)
                            # debug_image = output_image.copy()
                            output_image[r, c] = 255
                            # debug_image[r, c] = 127
                            # debug_image = cv2.resize(debug_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                            # cv2.imshow("img", debug_image)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            done = False

        # check if whole image is white
        if np.all(output_image == 255):
            # print('Erased edges flood fill failed, whole image is white')
            return image
        else:
            return output_image

        return output_image

    # helper method for ocr specifically just to detect 10's. because im fuckign tired
    def check_edge_transitions(self, image, fifth=2):
        height, width = image.shape
        row_to_check = height // 5 * fifth
        transition_count = 0

        for col in range(1, width):
            # check for a white-black transition or a black-white transition
            if image[row_to_check, col] != image[row_to_check, col - 1]:
                transition_count += 1

        # print(transition_count)
        return transition_count

    def find_black_pixel_height(self, image, tolerance=20):
        # Ensure the input is a 2D grayscale matrix
        if len(image.shape) != 2:
            raise ValueError("The input should be a two-dimensional grayscale image.")

        # Define a threshold to determine "black" based on the tolerance
        threshold = tolerance

        # Find the indices of rows containing pixels darker than the threshold
        # This means those pixels are considered "black" based on the tolerance
        black_pixel_rows = np.where(np.any(image <= threshold, axis=1))[0]

        # Check if there are any rows with such "black" pixels
        if len(black_pixel_rows) == 0:
            return 0  # Return 0 if no black pixel is found

        # Get the first and last row indices with tolerant black pixels
        first_black_row = black_pixel_rows[0]
        last_black_row = black_pixel_rows[-1]

        # Calculate the vertical height between the first and last black rows
        height_between = last_black_row - first_black_row

        return height_between


    def ocr_text_from_image(self, screenshot: np.ndarray,
                            location: tuple[int, int, int, int],
                            rotation_angle=0,
                            psm=7,
                            invert=False,
                            black_text=False,
                            erode=False,
                            brightness=0.0,
                            contrast=0.0,
                            card_chars=True,
                            similarity_factor=True,
                            scale=40):

        image = screenshot[location[1]:location[3], location[0]:location[2]]

        if rotation_angle != 0:
            image = Image.fromarray(image)
            image = image.rotate(rotation_angle, resample=Resampling.BICUBIC, fillcolor=(255, 255, 255))
            image = np.array(image)

        if invert:
            # white pixels matter more than colored, so decrease brightness of saturated pixels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image = np.float32(image)

            # scale the V value adjustments based on S value
            hsv_image[:, :, 2] = hsv_image[:, :, 2] * (1 + (-1.5 * (hsv_image[:, :, 1] / 255)))
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

            # scale down dark values
            hsv_image[:, :, 2] = np.where(hsv_image[:, :, 2] < 192, hsv_image[:, :, 2] * (hsv_image[:, :, 2] / 255), hsv_image[:, :, 2])

            # set any value above 192 to 255
            # hsv_image[:, :, 2] = np.where(hsv_image[:, :, 2] > 192, 255, hsv_image[:, :, 2])


            hsv_image = np.uint8(hsv_image)
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        if black_text:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image = np.float32(image)

            # scale up the V value adjustments based on S value
            hsv_image[:, :, 2] = hsv_image[:, :, 2] * (1 + (1.5 * (hsv_image[:, :, 1] / 255)))
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

            # set the S value to 0
            hsv_image[:, :, 1] = 0

            hsv_image[:, :, 2] = np.where(hsv_image[:, :, 2] > 45, 255, hsv_image[:, :, 2])

            hsv_image = np.uint8(hsv_image)
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


        if brightness > 0.0:
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

        if contrast > 0:
            img = Image.fromarray(np.uint8(image))
            contrasted_img = ImageEnhance.Contrast(img).enhance(contrast)
            image = np.array(contrasted_img)

        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if similarity_factor:
            center_pixel = image[image.shape[0] // 2, image.shape[1] // 2]
            distances = np.linalg.norm(image.astype(float) - center_pixel, axis=2)
            similarity_factor = np.sum(distances <= 7) / (image.shape[0] * image.shape[1])
            if similarity_factor > 0.94:
                print(f'Image is mostly the same color as center pixel, returning empty string {similarity_factor}')
                return ''

        # TODO this is occasionally failing. I don't know why
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale height to 33 pixels using bicubic resampling
        # invert the image
        scale_from = self.find_black_pixel_height(gray if not invert else cv2.bitwise_not(gray))
        # print(scale_from)
        if scale_from != 0:
            gray = cv2.resize(gray, (0, 0), fx=scale / scale_from, fy=scale / scale_from, interpolation=cv2.INTER_CUBIC)

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

      

        # cv2.imshow("img", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if invert:
            binary = cv2.bitwise_not(binary)

        binary = self.erase_edges(binary)
        binary = self.eliminate_isolated_pixels(binary)
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # cv2.imshow("img2", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if black_text and self.check_edge_transitions(binary) == 6:
            # print("looks like a ten")
            result = '10'
        else:
            if card_chars:
                config = CUSTOM_CONFIG.format(psm=psm)
            else:
                config = '--oem 3 --psm {psm}'.format(psm=psm)
            result: str = pytesseract.image_to_string(binary, lang='eng', config=config).strip()

        # print("result: \"" + result + "\"")

        # can you tell what number has given me hours of trouble with tesseract and TJ font
        # EDIT: numberS*
        if any(result == char for char in ['0', 'O', '1', 'I', "170", "I70", "17O", "I7O", "70", "1O", "IO", "I0", "7O", '']):
            if self.check_edge_transitions(binary) == 2 and self.check_edge_transitions(binary, 3) == 4 and any(result == char for char in ['0', 'O', '']):
                return '6'
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

    locations2 = poker_img_detect.find_community_suits(img, threshold=0.95)

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
