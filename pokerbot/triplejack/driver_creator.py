from selenium import webdriver


import pywinctl


def create_tj_driver(headless=False):

    options = webdriver.FirefoxOptions()

    # options.add_argument("--no-sandbox")
    # options.add_argument("--use-angle=vulkan")
    # options.add_argument("--enable-features=Vulkan")
    # options.add_argument("--disable-vulkan-surface")
    # options.add_argument("--enable-unsafe-webgpu")

    # options.add_argument("--mute-audio")

    # options.add_experimental_option("useAutomationExtension", False)
    # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.add_argument("--disable-infobars")
    # options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

    # use /usr/bin/google-chrome-stable
    # options.binary_location = "/usr/bin/google-chrome-stable"
    
    if headless:
        options.add_argument("--headless")
        # options.add_argument("--window-size=1280,720")
        driver = webdriver.Firefox(options=options)
        # driver.maximize_window() # why is this setting to 4k lol, no need
    else:
        # options.add_argument("--window-size=1280,720")
        # options.add_argument("--start-maximized")
        driver = webdriver.Firefox(options=options)


    driver.set_window_position(0,0)
    driver.set_window_size(1280, 720, windowHandle="current")
   

    return driver
