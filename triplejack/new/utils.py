import cv2
import numpy as np
def prepare_ss(ss: bytes, flags=cv2.IMREAD_UNCHANGED) -> cv2.typing.MatLike:
    return cv2.imdecode(np.frombuffer(ss, np.uint8), cv2.IMREAD_COLOR)