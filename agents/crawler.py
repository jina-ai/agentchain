import tempfile
import time
import uuid
from collections import deque
from bs4 import BeautifulSoup

import numpy as np
import openai
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import BytesIO
from PIL import Image

def get_ntokens(text, model='gpt-4'):
    import tiktoken
    enc = tiktoken.encoding_for_model(model)
    res = enc.encode(text)
    return len(res)

def image_vqa(image, question):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file)
        import replicate
        output = replicate.run(
            "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
            input={"image": open(temp_file.name, "rb"), "question": question}
        )
    return output

def remove_non_human_readable(soup):
    # Remove <style>, <script>, and <svg> tags
    for tag in soup(['style', 'script', 'svg']):
        tag.decompose()

    # Get the human-readable text
    return soup

def get_cleaned_html(element):
    html = element.get_attribute("innerHTML")
    soup = BeautifulSoup(html, "lxml")
    soup = remove_non_human_readable(soup)
    return soup.prettify()


def display_image(image):

    # Save the image to the temporary file
    image.save(f'image/{uuid.uuid4()}.png', "PNG")


def calculate_scaled_sub_image(original_image_size, scaled_image_size, sub_image_coord, sub_image_size):
    x, y = original_image_size
    z, w = scaled_image_size
    x1, y1 = sub_image_coord
    width, height = sub_image_size

    width_ratio = z / x
    height_ratio = w / y

    x1_scaled = x1 * width_ratio
    y1_scaled = y1 * height_ratio

    sub_image_width_scaled = width * width_ratio
    sub_image_height_scaled = height * height_ratio

    sub_image_coord_scaled = (x1_scaled, y1_scaled)
    sub_image_size_scaled = (sub_image_width_scaled, sub_image_height_scaled)

    return sub_image_coord_scaled, sub_image_size_scaled


def get_node_image(target_element):
    png = target_element.screenshot_as_png
    # # Capture a screenshot of the whole page
    # png = driver.get_screenshot_as_png()
    #
    # Crop the screenshot to the target element
    im = Image.open(BytesIO(png))
    # element_coordinates_scaled, element_size_scaled = calculate_scaled_sub_image(
    #     (root.size['width'], root.size['height']),
    #     im.size,
    #     (target_element.location['x'], target_element.location['y']),
    #     (target_element.size['width'], target_element.size['height']),
    # )
    #
    # left = element_coordinates_scaled[0]
    # top = element_coordinates_scaled[1]
    # right = element_coordinates_scaled[0] + element_size_scaled[0]
    # bottom = element_coordinates_scaled[1] + element_size_scaled[1]
    # im = im.crop((left, top, right, bottom))

    return im

def element_vqa(element, question='describe in detail what is in this image'):
    im = get_node_image(element)
    return image_vqa(im, question=question)

def is_useful(element, goal):
    prompt = f"""
You are a crawling expert. Your goal is the following: {goal}
You are given an html element, this is the description of the rendered element image: {element_vqa(element)}
This is the text data contained in this element: {get_cleaned_html(element)}.
If you think this element is useful for you to achieve your goal, reply by yes or no. Just say yes or no, nothing else
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0,
    )

    return response.choices[0].text.strip().lower() == 'yes'


# Set the path to the ChromeDriver executable
chromedriver_path = 'binaries/chromedriver_mac_arm64/chromedriver'

# Set up Chrome options
chrome_options = Options()
chrome_options.headless = True
# Create a new instance of the Chrome driver
driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options)


# Navigate to the desired website
url = "https://www.redfin.com"

driver.get(url)

# Get the page source and parse it with BeautifulSoup
# page_source = driver.page_source

# soup = BeautifulSoup(page_source, "html.parser")

initial_goal = "Extract houses to buy in houston that works for a family of 4, my budget is 600k"

root = driver.find_element(By.TAG_NAME, "html")
# driver.set_window_size(root.size['width'], root.size['height'])
total_width = driver.execute_script("return document.body.scrollWidth")
total_height = driver.execute_script("return document.body.scrollHeight")

# Set the window size to the total size of the web page
driver.set_window_size(total_width, total_height)

# Perform BFS traversal on the DOM nodes
queue = deque([root])
while queue:
    current_element = queue.popleft()
    if 5 <= get_ntokens(get_cleaned_html(current_element), model='text-davinci-003') < 1000 and len(current_element.text) > 30:
        if is_useful(current_element, goal=initial_goal):
            print(current_element.text)
            get_node_image(current_element).show()


    else:
        # Get child elements using xpath
        children = current_element.find_elements(By.XPATH, "./*")
        for child in children:
            queue.append(child)

# element = driver.find_element(By.ID, 'homepageTabContainer')
# element = driver.find_element(by=By.CSS_SELECTOR, value='[data-rf-test-id="homepage_main"]')
# im = get_node_image(root, element, driver)
# im.show()

# print(is_useful(root, driver.find_element(By.ID, 'homepageTabContainer'), driver, goal=initial_goal))

