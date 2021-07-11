import cv2
import numpy as np

# Reading the images
image_to_look = cv2.imread('image_to_look.png', cv2.IMREAD_UNCHANGED)
image_to_find = cv2.imread('image_to_find.png', cv2.IMREAD_UNCHANGED)

#Matching the images
result = cv2.matchTemplate(image_to_look, image_to_find, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get the wiht of the image_to_find
width = image_to_find.shape[1]

# Get the height of the image_to_find
height = image_to_find.shape[0]

# The percent mathc accept to confirm the image is really found
percent_match = .80

# Find withc matches with the 80%
yloc, xloc = np.where(result >= percent_match)

rectangles = []

# Mounting the rectangles to show in the image_to_look
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(width), int(height)])
    rectangles.append([int(x), int(y), int(width), int(height)])

# Grouping the rectangles
rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

# Show the rectangles in the image_to_look
for (x, y, width, height) in rectangles:
    cv2.rectangle(image_to_look, (x, y), (x + width, y + height), (0, 255, 255), 2)

# Open a window to show the results
cv2.imshow('Result', image_to_look)
cv2.waitKey()
cv2.destroyAllWindows()
