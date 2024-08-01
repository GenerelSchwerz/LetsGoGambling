import subprocess


import pywinctl as gw

from PIL import ImageGrab

import mss

import cv2
import numpy as np


def get_window_title(window_id):
    try:
        window_title = subprocess.check_output(
            ['xdotool', 'getwindowname', str(window_id)],
            universal_newlines=True
        ).strip()
        return window_title
    except subprocess.CalledProcessError as e:
      # print(f"Failed to get window title: {e}")
        return None

def get_all_windows_matching(title) -> list[int]:
    try:
        matching_windows = subprocess.check_output(
            ['xdotool', 'search', '--name', title],
            universal_newlines=True
        ).splitlines()  

        return list(map(int, matching_windows))
    except subprocess.CalledProcessError as e:
      # print(f"Failed to get windows matching '{title}': {e}")
        return []
    
def move_window_id_to_forefront(window_id):
    try:
        subprocess.run(
            ['xdotool', 'windowactivate', str(window_id)],
            check=True
        )
      # print(f"Moved window '{window_id}' to the front.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to move window to the front: {e}")
        pass

def get_window_id_dimensions(window_id):
    try:
        window_geometry = subprocess.check_output(
            ['xdotool', 'getwindowgeometry', '--shell', str(window_id)],
            universal_newlines=True
        )
      # print(window_geometry)

        # return (x, y, x + width, y + height)

        lines = window_geometry.splitlines()
        x = int(lines[1].split('=')[1])  
        y = int(lines[2].split('=')[1])
        
        width = int(lines[3].split('=')[1])
        height = int(lines[4].split('=')[1])

      # print(f"x: {x}, y: {y}, width: {width}, height: {height}")

        return (x, x + width, y, y + height)
    except subprocess.CalledProcessError as e:
        print(f"Failed to get window dimensions: {e}")
        return None, None

def screenshot_window_id(window_id):
    try:
        x1, x2, y1, y2 = get_window_id_dimensions(window_id)
        # with mss.mss() as sct:
        #   # print(sct.monitors)
        #     monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1, "mon": 0}
        #     sct_img = sct.grab(monitor)
        sct_img = ImageGrab.grab(bbox=(x1, y1, x2, y2), all_screens=True)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Failed to take screenshot: {e}")

def move_window_id_to_background(window_id):
    try:
        subprocess.run(
            ['xdotool', 'windowminimize', str(window_id)],
            check=True
        )
      # print(f"Moved window '{window_id}' to the background.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to move window to the background: {e}")
    
def move_window_id_to_display(window_id, desktop):
    try:
        subprocess.run(
            ['wmctrl', '-i', '-r', str(window_id), '-t', str(desktop)],
            check=True
        )
      # print(f"Moved window '{window_id}' to desktop {desktop}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to move window to display: {e}")

def resize_window_id(window_id, width, height):
    try:
        subprocess.run(
            ['xdotool', 'windowsize', str(window_id), str(width), str(height)],
            check=True
        )
      # print(f"Resized window '{window_id}' to {width}x{height}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to resize window: {e}")


def kill_window_id(window_id):
    try:
        subprocess.run(
            ['xdotool', 'windowkill', str(window_id)],
            check=True
        )
      # print(f"Killed window '{window_id}'.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill window: {e}")

def resize_window(window_title, width, height):
    try:

        all_windows = subprocess.check_output(['xdotool', 'search', '--name', '.*'], universal_newlines=True).strip().split('\n')
        for window in all_windows:
            title = subprocess.check_output(['xdotool', 'getwindowname', window], universal_newlines=True).strip()
            print(f"Window ID: {window}, Title: {title}")

        # Find the window ID using xdotool
        window_id = subprocess.check_output(
            ['xdotool', 'search', '--name', str(window_title)],
            universal_newlines=True
        ).splitlines()

        # get parent window id
        ints = [int(x) for x in window_id]
        window_id = str(max(ints))


        # Resize the window
        subprocess.run(
            ['xdotool', 'windowsize', window_id, str(width), str(height)],
            check=True
        )
        print(f"Resized window '{window_title} ({window_id})' to {width}x{height}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to resize window: {e}")


# gw.getActiveWindow()

if __name__ == "__main__":

 

    import time

    # for i in range(1):
    ids = get_all_windows_matching(".+NL.+")
    for id in ids:

        resize_window_id(id, 1080, 720)
        move_window_id_to_display(id, 0)
        move_window_id_to_forefront(id)
        time.sleep(0.2)
        print(get_window_id_dimensions(id))
        img = screenshot_window_id(id)

        cv2.imwrite(f"pokerstars_{id}-{0}.png", img)
        # move_window_id_to_background(id)


        time.sleep(2)
        

        

