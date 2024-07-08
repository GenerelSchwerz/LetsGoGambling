import cv2
import numpy as np
import pytesseract


#  stolen code
def non_max_suppression_slow(boxes, overlapThresh=0):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]



class PokerImgDetect:

    def __init__(self, imgs_path: str) -> None:
        self.imgs_path = imgs_path

        self.SIT_BUTTON_BYTES = None
        self.COMMUNITY_HEART_SUIT_BYTES = None
        self.COMMUNITY_DIAMONDS_SUIT_BYTES = None
        self.COMMUNITY_CLUBS_SUIT_BYTES = None
        self.COMMUNITY_SPADES_SUIT_BYTES = None

        


    @staticmethod
    def prepare_ss(screenshot: bytes):
        fullimg = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(fullimg, cv2.IMREAD_GRAYSCALE)
        # return cv2.threshold(img, 127, 255, 0)
        return img

    @staticmethod
    def magic_detection(fullimg: np.ndarray, wanted: np.ndarray, threshold=0.77):
        print(wanted.shape)
        # w, h = wanted.shape[:-1]
        w, h = wanted.shape[::]
        res = cv2.matchTemplate(fullimg, wanted, cv2.TM_CCOEFF_NORMED)
 
        loc = np.where(res >= threshold)

        zipped = np.array(
            [[pt[0], pt[1], pt[0] + h, pt[1] + w] for pt in zip(*loc[::-1])]
        )

        return non_max_suppression_slow(zipped, 0.3)
    

    def load_images(self):
        # greyscale images
        self.SIT_BUTTON_BYTES = cv2.imread(self.imgs_path + "sit.png", cv2.IMREAD_GRAYSCALE)


        self.COMMUNITY_HEART_SUIT_BYTES = cv2.imread(self.imgs_path + "heart.png", cv2.IMREAD_GRAYSCALE)
        self.COMMUNITY_DIAMONDS_SUIT_BYTES = cv2.imread(self.imgs_path + "diamond.png", cv2.IMREAD_GRAYSCALE)
        self.COMMUNITY_CLUBS_SUIT_BYTES = cv2.imread(self.imgs_path + "club.png", cv2.IMREAD_GRAYSCALE)
        self.COMMUNITY_SPADES_SUIT_BYTES = cv2.imread(self.imgs_path + "spade.png", cv2.IMREAD_GRAYSCALE)

        self.HOLE_HEART_SUIT_BYTES = cv2.imread(self.imgs_path + "hole_heart1.png", cv2.IMREAD_GRAYSCALE)
        self.HOLE_DIAMONDS_SUIT_BYTES = cv2.imread(self.imgs_path + "hole_diamond1.png", cv2.IMREAD_GRAYSCALE)
        self.HOLE_CLUBS_SUIT_BYTES = cv2.imread(self.imgs_path + "hole_club1.png", cv2.IMREAD_GRAYSCALE)
        self.HOLE_SPADES_SUIT_BYTES = cv2.imread(self.imgs_path + "hole_spade1.png", cv2.IMREAD_GRAYSCALE)

        _, self.COMMUNITY_CLUBS_SUIT_BYTES = cv2.threshold(self.COMMUNITY_CLUBS_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.COMMUNITY_DIAMONDS_SUIT_BYTES = cv2.threshold(self.COMMUNITY_DIAMONDS_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.COMMUNITY_HEART_SUIT_BYTES = cv2.threshold(self.COMMUNITY_HEART_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.COMMUNITY_SPADES_SUIT_BYTES = cv2.threshold(self.COMMUNITY_SPADES_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)

        _, self.HOLE_CLUBS_SUIT_BYTES = cv2.threshold(self.HOLE_CLUBS_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.HOLE_DIAMONDS_SUIT_BYTES = cv2.threshold(self.HOLE_DIAMONDS_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.HOLE_HEART_SUIT_BYTES = cv2.threshold(self.HOLE_HEART_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)
        _, self.HOLE_SPADES_SUIT_BYTES = cv2.threshold(self.HOLE_SPADES_SUIT_BYTES, 127, 255, cv2.THRESH_BINARY)

        # self.COMMUNITY_HEART_SUIT_BYTES = cv2.Canny(self.COMMUNITY_HEART_SUIT_BYTES, 100, 200)
        # self.COMMUNITY_DIAMONDS_SUIT_BYTES = cv2.Canny(self.COMMUNITY_DIAMONDS_SUIT_BYTES, 100, 200)
        # self.COMMUNITY_CLUBS_SUIT_BYTES = cv2.Canny(self.COMMUNITY_CLUBS_SUIT_BYTES, 100, 200)
        # self.COMMUNITY_SPADES_SUIT_BYTES = cv2.Canny(self.COMMUNITY_SPADES_SUIT_BYTES, 100, 200)

        # self.HOLE_HEART_SUIT_BYTES = cv2.Canny(self.HOLE_HEART_SUIT_BYTES, 100, 200)
        # self.HOLE_DIAMONDS_SUIT_BYTES = cv2.Canny(self.HOLE_DIAMONDS_SUIT_BYTES, 100, 200)
        # self.HOLE_CLUBS_SUIT_BYTES = cv2.Canny(self.HOLE_CLUBS_SUIT_BYTES, 100, 200)
        # self.HOLE_SPADES_SUIT_BYTES = cv2.Canny(self.HOLE_SPADES_SUIT_BYTES, 100, 200)

        # self.COMMUNITY_HEART_SUIT_CONTOURS,_ = cv2.findContours(self.COMMUNITY_HEART_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.COMMUNITY_DIAMONDS_SUIT_CONTOURS,_ = cv2.findContours(self.COMMUNITY_DIAMONDS_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.COMMUNITY_CLUBS_SUIT_CONTOURS,_ = cv2.findContours(self.COMMUNITY_CLUBS_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.COMMUNITY_SPADES_SUIT_CONTOURS ,_ = cv2.findContours(self.COMMUNITY_SPADES_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # self.HOLE_HEART_SUIT_CONTOURS ,_ = cv2.findContours(self.HOLE_HEART_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.HOLE_DIAMONDS_SUIT_CONTOURS ,_ = cv2.findContours(self.HOLE_DIAMONDS_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.HOLE_CLUBS_SUIT_CONTOURS ,_ =cv2.findContours(self.HOLE_CLUBS_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.HOLE_SPADES_SUIT_CONTOURS ,_ =cv2.findContours(self.HOLE_SPADES_SUIT_BYTES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # cv2.imshow("hearts_community", self.COMMUNITY_HEART_SUIT_BYTES)
        # cv2.imshow("diamonds_community", self.COMMUNITY_DIAMONDS_SUIT_BYTES)
        # cv2.imshow("clubs_community", self.COMMUNITY_CLUBS_SUIT_BYTES)
        # cv2.imshow("spades_community", self.COMMUNITY_SPADES_SUIT_BYTES)

        # cv2.imshow("hearts_hole", self.HOLE_HEART_SUIT_BYTES)
        # cv2.imshow("diamonds_hole", self.HOLE_DIAMONDS_SUIT_BYTES)
        # cv2.imshow("clubs_hole", self.HOLE_CLUBS_SUIT_BYTES)
        # cv2.imshow("spades_hole", self.HOLE_SPADES_SUIT_BYTES)

        # cv2.waitKey(0)
        # import time
        # time.sleep(3)
        # cv2.destroyAllWindows()




    def detect_sit_button(self, screenshot: cv2.typing.MatLike, threshold=0.77):
        return self.magic_detection(screenshot, self.SIT_BUTTON_BYTES, threshold=threshold)
    
    def community_suits(self, ss1: cv2.typing.MatLike, threshold=0.77):
        # _, ss1 = cv2.threshold(ss1, 127, 255, cv2.THRESH_BINARY)
        hearts = self.magic_detection(ss1, self.COMMUNITY_HEART_SUIT_BYTES, threshold=threshold)
        diamonds = self.magic_detection(ss1, self.COMMUNITY_DIAMONDS_SUIT_BYTES, threshold=threshold)
        clubs = self.magic_detection(ss1, self.COMMUNITY_CLUBS_SUIT_BYTES, threshold=threshold)
        spades = self.magic_detection(ss1, self.COMMUNITY_SPADES_SUIT_BYTES, threshold=threshold)
        return {
            "hearts": hearts,
            "diamonds": diamonds,
            "clubs": clubs,
            "spades": spades
        }
        
    def hole_suits(self, screenshot: cv2.typing.MatLike, threshold=0.85):
        # _, screenshot = cv2.threshold(screenshot, 127, 255, cv2.THRESH_BINARY)
        hearts = self.magic_detection(screenshot, self.HOLE_HEART_SUIT_BYTES, threshold=threshold)
        diamonds = self.magic_detection(screenshot, self.HOLE_DIAMONDS_SUIT_BYTES, threshold=threshold)
        clubs = self.magic_detection(screenshot, self.HOLE_CLUBS_SUIT_BYTES, threshold=threshold)
        spades = self.magic_detection(screenshot, self.HOLE_SPADES_SUIT_BYTES, threshold=threshold)
        return {
            "hearts": hearts,
            "diamonds": diamonds,
            "clubs": clubs,
            "spades": spades
        }
    
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        print(angle)
        return image
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


    def card_number(self, screenshot: cv2.typing.MatLike, suit_bb: tuple[int, int, int, int], threshold=0.77):
        w, h = suit_bb[2] - suit_bb[0], suit_bb[3] - suit_bb[1]

        def setup_img(ss):
              # Threshold to create a binary image
            _, binary = cv2.threshold(subsection, 1, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask of the same size as the image, initialized to white (255)
            mask = np.ones_like(subsection) * 255
            return mask, contours
        
        subsection = screenshot[suit_bb[1] - h - h // 6: suit_bb[1], suit_bb[0] - w // 4: suit_bb[2] + w // 4]

        # area of subsectio
        tot_area = w * h
        mask, contours = setup_img(subsection)

        # Get image dimensions
        height, width = subsection.shape

        def ident_valid_contours(contours, max_h, max_w):

            # Draw only the contours that do not touch the edge
            for contour in contours:

                # eliminate contours that:
                # touch the edge of the image AND run across an entire side of the image

                x, y, w, h = cv2.boundingRect(contour)
                if x == 0 or y == 0 or x + w == max_w or y + h == max_h:
                    continue

                # eliminate contours that:
                # are too small

                if cv2.contourArea(contour) < tot_area // 10:
                    continue

                yield contour



           
        good_cnts = list(ident_valid_contours(contours, width, height))
        # default
        print("tot area per")
        if len(good_cnts) == 0:
            cv2.drawContours(mask, contours, -1, (0), thickness=cv2.FILLED)
        else:
            cv2.drawContours(mask, good_cnts, -1, (0), thickness=cv2.FILLED)

        # Invert the mask to get black text on white background
        mask = cv2.bitwise_not(mask)

        subsection = cv2.bitwise_not(subsection)
        # Combine the mask with the original image
        result = cv2.bitwise_and(subsection, subsection, mask=mask)
        result = cv2.bitwise_not(result)

        # pad the image with white pixels, crop to 33x33
        result = cv2.copyMakeBorder(result, height // 6, height // 6, width // 4, width //4, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # resize to 33x33
        result = cv2.resize(result, (33, 33))
   
        # invert
        # subsection = cv2.bitwise_not(subsection)

        subsection = self.deskew(subsection)

        # configure to read only poker card numbers


        text = pytesseract.image_to_string(result, config='--psm 10 -c tessedit_char_whitelist=0123456789AKQJ', lang='eng').strip()
        text = text if text != "0" else "10"


        cv2.imshow(f"Found {text} mask", mask)
        cv2.imshow(f"Found {text} res", result)
        cv2.imshow(f"Found {text}", subsection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print (text, text == "0")
        return text
    

if __name__ == "__main__":
    poker_img_detect = PokerImgDetect("tests/")
    poker_img_detect.load_images()

    org_img = open("tests/sitting_real.png", "rb").read()
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

    locations2 = poker_img_detect.community_suits(img, threshold=0.77)

    for key, locations in locations2.items():
        for pt in locations:
            print("community", key, pt)
            cv2.rectangle(img2, (pt[0], pt[1]), (pt[2], pt[3]), (0, 255, 0), 2)
            poker_img_detect.card_number(img, pt)

    locations3 = poker_img_detect.hole_suits(img, threshold=0.85)

    for key, locations in locations3.items():
        for pt in locations:
            print("hole", key, pt)
            cv2.rectangle(img2, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 2)

            poker_img_detect.card_number(img2, pt)


    cv2.imwrite("result.png", img2)
    cv2.imwrite("used.png", img)
    pass
