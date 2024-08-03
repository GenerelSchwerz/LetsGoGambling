from .windowManager import AWindowManager

import os 

if os.name == "nt":
    from .windowManager import WindowsWindowManager
else:
    from .windowManager import UnixWindowManager
from .xdotoolUtils import get_all_windows_matching