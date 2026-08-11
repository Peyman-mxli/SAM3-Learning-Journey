"""
02-opencv-debugging.py

Example from the Agentic AI Programming session.

Goal:
Practice the AI-assisted debugging workflow using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Create an empty black image
image = np.zeros(
    (300, 300, 3),
    dtype=np.uint8
)


# --------------------------------------------------
# Example of a problematic implementation
# --------------------------------------------------

# OpenCV drawing coordinates should be integer values.
# Using floating-point coordinates can produce an error
# depending on the function and OpenCV version.

point1 = (50.5, 50.5)
point2 = (250.5, 250.5)


# --------------------------------------------------
# Debugging
# --------------------------------------------------

# Convert the coordinates to integers.

point1_fixed = (
    int(point1[0]),
    int(point1[1])
)

point2_fixed = (
    int(point2[0]),
    int(point2[1])
)


# --------------------------------------------------
# Correct implementation
# --------------------------------------------------

cv2.rectangle(
    image,
    point1_fixed,
    point2_fixed,
    (255, 0, 0),
    3
)


# --------------------------------------------------
# Display the result
# --------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("OpenCV Debugging Example")
plt.axis("off")
plt.show()


# --------------------------------------------------
# Agentic debugging workflow
# --------------------------------------------------

"""
The workflow practiced in this example is:

1. Generate or write code.
2. Execute the code.
3. Read the error message.
4. Provide the error to the AI assistant.
5. Ask the AI to explain the problem.
6. Review the proposed solution.
7. Correct the code.
8. Execute it again.

The developer remains responsible for reviewing
and validating the AI-generated solution.
"""
