import cv2
import numpy as np

img = cv2.imread('images/img_20240114200551.png')
print(img.shape)
square_side_length = min(img.shape[0], img.shape[1])
img = img[0:square_side_length, 0:square_side_length]
cv2.imwrite('img_20240114200551_sq.png', img)
