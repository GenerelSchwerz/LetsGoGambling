import cv2
import numpy as np

def prepare_ss(ss: bytes, flags=cv2.IMREAD_UNCHANGED) -> cv2.typing.MatLike:
    img = cv2.imdecode(np.frombuffer(ss, np.uint8), flags=flags)

    # if img is bgra, convert to bgr
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img