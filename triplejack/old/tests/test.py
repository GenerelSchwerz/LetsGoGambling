import cv2
import numpy as np

# Function to check if a contour is likely a card
def is_card(contour, min_area=1000, max_area=10000, min_aspect_ratio=1.5, max_aspect_ratio=2.5):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate area and aspect ratio
    area = w * h
    aspect_ratio = float(w) / h if h != 0 else 0
    
    print(area)
    
    # Check if the contour meets the criteria
    if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
        return True
    return False

# Step 1: Read the image
image = cv2.imread('sitting_real.png')

# Step 2: Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, edged = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 50, 200)

# Step 3: Find contours
contours, _ = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

# Step 4: Filter and approximate contours
card_contours = []
for contour in contours:
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if the approximated contour has four points (a rectangle) and meets the card criteria
    if len(approx) == 4:
        card_contours.append(approx)

# Step 5: Draw bounding rectangles
for contour in card_contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

# Display the result
cv2.imshow('Original Image', edged)
cv2.imshow('Detected Cards', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
