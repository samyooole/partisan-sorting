from selenium import webdriver
from selenium.webdriver.common.by import By
import time

DRIVER_PATH = '/path/to/chromedriver'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
driver.get('https://dl.ncsbe.gov/index.html?prefix=data/Snapshots/')

h1 = driver.find_elements(By.TAG_NAME, "a")


iterate = list(range(3, len(h1) + 1)) # 3 is when the voter files start
iterate = list(range(36, len(h1) + 1)) # 36 is when i timed out
iterate = list(range(51, len(h1) + 1)) # 51 is when i timed out
iterate = list(range(57, len(h1))) # 57 is when i timed out

for n in iterate:
    h1[n].click()
    time.sleep(60)
