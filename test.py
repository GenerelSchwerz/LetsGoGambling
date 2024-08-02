from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize the WebDriver (ensure you have the correct driver for your browser)
driver = webdriver.Chrome()

# Open the website
driver.get('https://www.replaypoker.com')

# Click the button that opens the new window
try:
    time.sleep(20)
except KeyboardInterrupt:
    pass



# <a href="/quick_play/13880796" class="btn btn-warning btn-play-now" data-bypass="true">Play Now</a>
button = driver.find_element(By.XPATH, '//a[@class="btn btn-warning btn-play-now"]')
button.click()

# Wait for the new window to open
time.sleep(2)  # Adjust as needed

# Get a list of all window handles
window_handles = driver.window_handles

# Switch to the new window (assuming the new window is the last opened)
new_window_handle = window_handles[-1]
driver.switch_to.window(new_window_handle)
    
# Perform actions in the new window
# For example, print the title of the new window
print(driver.title)

# Close the new window and switch back to the original
driver.close()
driver.switch_to.window(window_handles[0])

# Continue with further actions
driver.quit()
