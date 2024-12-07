'''The MIT License (MIT)

Copyright (c) 2017 Dhanushka Dangampola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''
import cv2
import numpy as np
import pytesseract
import os
import sys

# Set the Tesseract path (!!! Must be adjusted for each user !!!)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create 'crops' folder (ignore if it already exists)
if not os.path.exists('crops'):
    os.makedirs('crops')

image_file = sys.argv[1]
rgb = cv2.imread(image_file)
if rgb is None:
    print("NO image")
    sys.exit(1)
    
#Image Preprocessing    
img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)

index = 0 
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    padding = 8
    if r > 0.45 and w > 8 and h > 8:
       
        x_pad = max(0, x - padding)  
        y_pad = max(0, y - padding)  
        x2_pad = min(rgb.shape[1], x + w + padding)  
        y2_pad = min(rgb.shape[0], y + h + padding)  

    
        cropped = rgb[y_pad:y2_pad, x_pad:x2_pad]
        
        #Crop image Preprocessing  
        crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        
        _, crop_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        resized = cv2.resize(crop_bin, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # Save cropped image
        cropped_filename = f"crops/cropped_{index}.jpg"
        cv2.imwrite(cropped_filename, resized)
        
        # Perform OCR on the cropped image
        custom_config = r'--oem 3 --psm 8'
        text = pytesseract.image_to_string(resized, lang='eng+kor', config=custom_config)
        
        if text.strip():
            print(f"Text from cropped_{index}: {text.strip()}")
        
        index += 1

cv2.imshow('rects', rgb)
cv2.waitKey()
cv2.destroyAllWindows()
