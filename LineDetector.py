import matplotlib.pylab as plt
import cv2
import numpy as np

# Import airport runway image
image = cv2.imread('pista2.jpg')

# Definition of image dimensions
height = image.shape[0]
width = image.shape[1]

# Definition of region of interest (triangle vertices)
ROI_vertices = [
    (0,500),
    (330, 160),
    (width, 500)
]


# Function to extract the ROI from input image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask = 255
    cv2.fillPoly(mask, vertices, match_mask)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# Function that overlays lines to the original image
def represent_lines(img, lines, color=[255, 0, 0], thickness=3):
    if lines is None:
        return
    img = np.copy(img)
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            img.shape[2]
        ),
        dtype=np.uint8,
    )
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


# Returns Canny Edge Detection to extract edges of the image
canny_image = cv2.Canny(image, 100, 200)

# Returns Canny Edge Detection only applied to ROI
masked_image = region_of_interest(canny_image,
               np.array([ROI_vertices], np.int32))

# Hough Line Transform to detect lines in Canny ROI image
lines = cv2.HoughLinesP(
    masked_image,
    rho=30,
    theta=np.pi / 60,
    threshold=80,
    lines=np.array([]),
    minLineLength=180,
    maxLineGap=25
)

# Function to represent the lines obtained on the original image
line_image = represent_lines(image, lines)


# Plot
plt.figure()
plt.imshow(line_image)
plt.show()