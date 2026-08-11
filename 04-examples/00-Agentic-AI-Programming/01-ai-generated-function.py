"""
01-ai-generated-function.py

Example from the Agentic AI Programming session.

Goal:
Download an image from a URL, load it with OpenCV,
convert it to grayscale, and return the processed image.
"""

import urllib.request

import cv2
import numpy as np
import matplotlib.pyplot as plt


def descargar_y_procesar(url):
    """
    Download an image from a URL and convert it to grayscale.
    """

    # Download image into memory
    response = urllib.request.urlopen(url)
    image_data = response.read()

    # Convert downloaded bytes into a NumPy array
    image_array = np.asarray(
        bytearray(image_data),
        dtype=np.uint8
    )

    # Decode the image with OpenCV
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "The image could not be decoded."
        )

    # Convert BGR image to grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale


# Example image
image_url = "https://ultralytics.com/images/bus.jpg"

# Process image
gray_image = descargar_y_procesar(image_url)

# Display result
plt.figure(figsize=(10, 6))
plt.imshow(gray_image, cmap="gray")
plt.title("AI-Generated Image Processing Example")
plt.axis("off")
plt.show()
