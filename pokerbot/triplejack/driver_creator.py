from selenium import webdriver
from selenium.webdriver.firefox.service import Service

def create_tj_driver(headless=False, firefox=False):

    if firefox:
        options = webdriver.FirefoxOptions()
        ser: Service = Service(r"C:\PATH Programs\geckodriver.exe")
        options.set_preference("media.volume_scale", "0.0")
        options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"


        if headless:
            options.add_argument("--headless")
            driver = webdriver.Firefox(service = ser, options=options)
        else:
            driver = webdriver.Firefox(options=options)

    else:

        # options = webdriver.FirefoxOptions()
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
        # options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

     
        if headless:
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)
        else:
            driver = webdriver.Chrome(options=options)

    driver.set_window_position(0, 0)
    driver.set_window_size(1886, 1056, windowHandle="current")

    return driver
