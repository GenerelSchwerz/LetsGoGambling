import cv2
import pytesseract
import numpy as np

import os

# Path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as necessary

def detect_thin_green_rectangle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # Use a combination of thresholding and morphological operations for better text detection
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show = np.zeros_like(image)
    cv2.drawContours(show, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    annotated_image = image.copy()
    names_and_locations = []
    
    # Loop over contours and annotate names
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < w < 600 and 15 < h < 50:  # Filtering based on the expected size of text boxes
            roi = gray[y:y+h, x:x+w]
            cv2.imshow('ROI', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            text = pytesseract.image_to_string(roi, config='--psm 7').strip()
            if text:
                names_and_locations.append((text, (x, y, w, h)))
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated_image, names_and_locations

folder = "/home/generel/Documents/code/python/poker/LetsGoGambling/triplejack/new/base/tests"

for filename in os.listdir(folder):
    if not filename.endswith('.png') and not filename.endswith('.jpg'):
        continue
    path = os.path.join(folder, filename)
    img, name_and_loc = detect_thin_green_rectangle(path)
    cv2.imshow('Annotated Image', img)
    print(f'File: {filename}')
    for name, loc in name_and_loc:
        print(f'Name: {name}, Location: {loc}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()