"""
01-image-loading.py

Introduction to Supervision — Example 01

Goal:
Download an example image, load it with OpenCV,
inspect its dimensions, convert BGR to RGB,
and display it with Matplotlib.
"""

import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Create the assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Download the example image
# --------------------------------------------------

image_url = "https://ultralytics.com/images/bus.jpg"
image_path = "assets/bus.jpg"

urllib.request.urlretrieve(
    image_url,
    image_path
)


# --------------------------------------------------
# 3. Load the image with OpenCV
# --------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# --------------------------------------------------
# 4. Inspect the image
# --------------------------------------------------

print("Image loaded successfully.")
print(f"Image shape: {image.shape}")

height, width, channels = image.shape

print(f"Height: {height} px")
print(f"Width: {width} px")
print(f"Channels: {channels}")


# --------------------------------------------------
# 5. Convert BGR to RGB
# --------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 6. Display the image
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(image_rgb)

plt.title("Example Image — OpenCV")
plt.axis("off")

plt.show()
