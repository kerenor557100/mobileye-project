import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

red_pixel = []
test = [[0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 52], [0, 53], [0, 54], [0, 55], [0, 59], [0, 60], [0, 61], [0, 62]]
img = Image.open("../images/berlin_000522_000019_leftImg8bit.png")

width, height = img.size
print(width, height)
for px in range(height):
    for py in range(width):
        [r, g, b] = img.getpixel((py, px))
        if r > g and r > b:
            red_pixel.append([px, py])

for i in red_pixel:
    cv2.putText(img, "X", i, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=  (255,0,0))

print(red_pixel)
plt.imshow(img)
plt.show()