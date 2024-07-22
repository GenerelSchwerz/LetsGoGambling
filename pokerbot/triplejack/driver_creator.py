from selenium import webdriver

def create_tj_driver(headless=False):

    options = webdriver.ChromeOptions()

    options.add_argument("--no-sandbox")
    options.add_argument("--use-angle=vulkan")
    options.add_argument("--enable-features=Vulkan")
    options.add_argument("--disable-vulkan-surface")
    options.add_argument("--enable-unsafe-webgpu")

    options.add_argument("--mute-audio")

    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("--disable-infobars")

    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        # driver.maximize_window() # why is this setting to 4k lol, no need
    else:
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)

    driver.set_window_size(1920, 1080)

    return driver
