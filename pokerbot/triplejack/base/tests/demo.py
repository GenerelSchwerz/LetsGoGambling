import cv2
import numpy as np
import os
import pytesseract


def display_and_wait(image, text="Image"):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clearly_define_cards(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # Use a combination of thresholding and morphological operations for better text detection
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    return binary



def __is_text_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return 20 < w < 1000 and 5 < h < 50

def clearly_define_text_boundaries(image, is_text_box=__is_text_box):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # Use a combination of thresholding and morphological operations for better text detection
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    ret = np.zeros_like(image)
    cv2.drawContours(ret, contours, -1, (0, 255, 0), 2)

    # Loop over contours and annotate names

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if __is_text_box(contour):
            cv2.rectangle(ret, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return ret

def test(image):
        # Load necessary libraries
    import cv2
    import numpy as np
    from PIL import Image

        # Define a tolerance value for slight color variations
    tolerance = 0

    # Create a mask where pixels with approximately equal R, G, and B values are white (255) and others are black (0)
    # Calculate the absolute differences between R, G, and B
    diff_rg = np.abs(image[:, :, 0] - image[:, :, 1])
    diff_rb = np.abs(image[:, :, 0] - image[:, :, 2])
    diff_gb = np.abs(image[:, :, 1] - image[:, :, 2])

    # Create a mask based on the tolerance
    mask = (diff_rg <= tolerance) & (diff_rb <= tolerance) & (diff_gb <= tolerance)

    # Convert the boolean mask to uint8 (0 or 255)
    mask = mask.astype(np.uint8) * 255
    # Apply the mask to the original image

    test = np.zeros_like(image)
    cv2.bitwise_and(image, image, mask=mask, dst=test)

   
    # Set all non-gray pixels to black
    test[mask == 0] = [127, 127, 127]
    # image[mask != 0] = [0,0,0]
 
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    
    # final = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    # final = cv2.dilate(gray, np.ones((1, 1), np.uint8), iterations=1)
    final = cv2.bitwise_not(gray)

    final[mask == 0] = 255

    # Load the new image using OpenCV
    new_image_cv = final

    # Apply binary thresholding
    _, binary_image = cv2.threshold(new_image_cv, 128, 255, cv2.THRESH_BINARY_INV)

    # Define a horizontal kernel for morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    # Detect horizontal lines using morphological operations
    detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Subtract the detected lines from the binary image to isolate text
    text_without_lines = cv2.subtract(binary_image, detected_lines)

    # Invert the image back to original colors
    text_without_lines = cv2.bitwise_not(text_without_lines)

    test = text_without_lines.copy()


    # Save the processed image for review


    custom_config = r'--oem 3 --psm 6'

    # Use pytesseract to extract text from the processed image
    extracted_text_processed = pytesseract.image_to_string(text_without_lines, config=custom_config)

    print(extracted_text_processed)
    cv2.imshow(f'Processed Image: {extracted_text_processed}', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def show_text_locations(image):
    # Define a tolerance value for slight color variations
    tolerance = 0

    # Create a mask where pixels with approximately equal R, G, and B values are white (255) and others are black (0)
    # Calculate the absolute differences between R, G, and B
    diff_rg = np.abs(image[:, :, 0] - image[:, :, 1])
    diff_rb = np.abs(image[:, :, 0] - image[:, :, 2])
    diff_gb = np.abs(image[:, :, 1] - image[:, :, 2])

    # Create a mask based on the tolerance
    mask = (diff_rg <= tolerance) & (diff_rb <= tolerance) & (diff_gb <= tolerance)

    # Convert the boolean mask to uint8 (0 or 255)
    mask = mask.astype(np.uint8) * 255
    # Apply the mask to the original image

    test = np.zeros_like(image)
    cv2.bitwise_and(image, image, mask=mask, dst=test)

   
    # Set all non-gray pixels to black
    test[mask == 0] = [127, 127, 127]
    # image[mask != 0] = [0,0,0]
 
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    
    # final = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    # final = cv2.dilate(gray, np.ones((1, 1), np.uint8), iterations=1)
    final = cv2.bitwise_not(gray)

    final[mask == 0] = 255

    # find contours
    contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    test1 = np.zeros_like(image)
    cv2.drawContours(test1, contours, -1, (0, 255, 0), 2)

    cv2.imshow('final', final)
    cv2.imshow('contours', test1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_data(final, config=config, output_type=pytesseract.Output.DICT)

    # Loop over each of the text components
    for i in range(len(text['text'])):
        if int(text['conf'][i]) > 60:  # Filter out weak confidence text
            (x, y, w, h) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Detected Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print detected player names and their positions
    for i in range(len(text['text'])):
        if int(text['conf'][i]) > 60:
            print(f"Detected name: {text['text'][i]} at position: (x: {text['left'][i]}, y: {text['top'][i]}, width: {text['width'][i]}, height: {text['height'][i]})")


folder = "/home/generel/Documents/code/python/poker/LetsGoGambling/triplejack/new/base/tests"


for filename in sorted(os.listdir(folder)):
    if not filename.endswith('.png') and not filename.endswith('.jpg'):
        continue
    path = os.path.join(folder, filename)
    img = cv2.imread(path)

    print(f'File: {filename}')

    display_and_wait(img)

    binary = clearly_define_cards(img)
 
    # display_and_wait(binary, "Binary")

    test(img)

    test1 = clearly_define_text_boundaries(img)

    # display_and_wait(test)

    show_text_locations(img)