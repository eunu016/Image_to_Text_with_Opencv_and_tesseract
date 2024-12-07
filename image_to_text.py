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

# Tesseract 경로 설정(사용자 마다 다름)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 'crops' 폴더 생성 (이미 존재하면 무시)
if not os.path.exists('crops'):
    os.makedirs('crops')

# 이미지 로드
image_file = sys.argv[1]
rgb = cv2.imread(image_file)
if rgb is None:
    print("NO image")
    sys.exit(1)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# 모폴로지 그라디언트
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

# 이진화 (이진화 + Otsu thresholding)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 커넥션 클로징 (텍스트 연결을 강화)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# 외곽선 찾기
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 마스크 생성
mask = np.zeros(bw.shape, dtype=np.uint8)

# 크롭된 영역을 텍스트로 인식하기 위한 처리
index = 0  # 크롭된 이미지 번호
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    padding = 8
    # 텍스트 영역으로 판단될 조건
    if r > 0.45 and w > 8 and h > 8:
        # 여백 추가된 크롭 영역 계산
        # 여백 추가된 크롭 영역 계산
        x_pad = max(0, x - padding)  # 왼쪽 경계
        y_pad = max(0, y - padding)  # 위쪽 경계
        x2_pad = min(rgb.shape[1], x + w + padding)  # 오른쪽 경계
        y2_pad = min(rgb.shape[0], y + h + padding)  # 아래쪽 경계

# 여백을 추가한 크롭된 이미지 저장
        cropped = rgb[y_pad:y2_pad, x_pad:x2_pad]


        # 크롭된 이미지 저장 (crops 폴더에 저장)
        crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        
        _, crop_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        resized = cv2.resize(crop_bin, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)


        cropped_filename = f"crops/cropped_{index}.jpg"
        cv2.imwrite(cropped_filename, resized)
        
        # 텍스트 인식
        custom_config = r'--oem 3 --psm 8'
        text = pytesseract.image_to_string(resized, lang='eng+kor', config=custom_config)
        
        # 공백인 텍스트는 출력하지 않음
        if text.strip():
            print(f"Text from cropped_{index}: {text.strip()}")
        
        index += 1

# 결과 이미지 출력 (원하는 경우)
cv2.imshow('rects', rgb)

cv2.waitKey()
cv2.destroyAllWindows()