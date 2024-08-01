"""
TODO: Implement window detection class.

Get the window size and position of the poker window.

Get the full title of the window.
"""

import cv2
import time

from abc import ABC, abstractmethod
class AWindowManager(ABC):
    def __init__(self, id: int = None, title: str = None):
        if id is None and title is None:
            raise RuntimeError("Must provide either a window id or title.")
        
        if id is not None and title is not None:
            raise RuntimeError("Must provide either a window id or title, not both.")
        
        if id is not None:
            self.assign_to_window(id)

        if title is not None:
            self.assign_to_window_title(title)


    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def assign_to_window(self, window_id: int):
        pass

    @abstractmethod
    def assign_to_window_title(self, window_title: str):
        pass

    @abstractmethod
    def get_window_dimensions(self) -> tuple[int, int, int, int]:
        pass

    @abstractmethod
    def update_window_title(self) -> str:
        pass

    @abstractmethod
    def ss(self) -> cv2.typing.MatLike:
        pass

    @abstractmethod
    def close(self):
        pass


class WindowsWindowManager(AWindowManager):
    def __init__(self):
        super().__init__()

    def assign_to_window(self, window_id: int):
        pass

    def assign_to_window_title(self, window_title: str):
        pass

    def get_window_dimensions(self) -> tuple[int, int, int, int]:
        pass

    def update_window_title(self) -> str:
        pass






    # Example usage
    # resize_window(".+No Limit Hold'em.+", 1920, 1080)  # Replace with the actual window title

from .xdotoolUtils import *

class UnixWindowManager(AWindowManager):
    """
    TODO: Make this actually work on Unix systems.

    Check for xdotool.
    """
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1080

    def is_valid(self) -> bool:
        try:
            self.update_window_title()
            return True
        except RuntimeError:
            return False
        

    def assign_to_window(self, window_id: int):
        title = get_window_title(window_id)
        if title is None:
            raise RuntimeError("Could not find window with id: " + str(window_id))
        
        self.window_id = window_id
        self.window_title = title

        resize_window_id(self.window_id, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

    def assign_to_window_title(self, window_title: str):
        window_id = get_all_windows_matching(window_title)

        if self.window_id is None:
            raise RuntimeError("Could not find window with title: " + window_title)
        
        if len(self.window_id) > 1:
            raise RuntimeError("Found more than one window with title: " + window_title)

        self.window_id = window_id[0]
        self.window_title = window_title

        resize_window_id(self.window_id, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

    def get_window_dimensions(self) -> tuple[int, int, int, int]:
        return get_window_id_dimensions(self.window_id)

    def update_window_title(self) -> str:
        self.window_title = get_window_title(self.window_id)
        if self.window_title is None:
            raise RuntimeError("Could not find window with id (most likely invalid): " + str(self.window_id))
        return self.window_title
    
    def close(self):
        kill_window_id(self.window_id)

    def ss(self, time_delay: int = 0.2):
        """
        TODO make this not ugly. Don't take up space on computer.
        """
        
        move_window_id_to_forefront(self.window_id)
        time.sleep(time_delay)
        ss = screenshot_window_id(self.window_id)

        # move_window_id_to_background(self.window_id)
        return ss

