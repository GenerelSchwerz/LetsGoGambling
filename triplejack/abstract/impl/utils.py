
ALLOWED_CHARS = "0123456789,.kMAKQJO"
CUSTOM_CONFIG = '--oem 3 --psm {psm} -c tessedit_char_whitelist=' + ALLOWED_CHARS

from dataclasses import dataclass

import numpy as np

@dataclass
class PokerImgOpts:
    # (image_path, binary)
    folder_path: str

    sit_button: (str, bool)

    community_hearts: (str, bool)
    community_diamonds: (str, bool)
    community_clubs: (str, bool)
    community_spades: (str, bool)

    hole_hearts: (str, bool)
    hole_diamonds: (str, bool)
    hole_clubs: (str, bool)
    hole_spades: (str, bool)

    check_button: (str, bool)
    call_button: (str, bool)
    bet_button: (str, bool)
    fold_button: (str, bool)
    raise_button: (str, bool)
    allin_button: (str, bool)


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

