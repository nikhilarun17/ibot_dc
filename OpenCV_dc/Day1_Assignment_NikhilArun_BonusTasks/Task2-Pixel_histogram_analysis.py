import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.png')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

flat=img_gray.flatten()
plt.hist(flat, bins=256, range=[0,256], color='red', alpha=0.7)
plt.title('Pixel Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(axis='y', alpha=0.75)
plt.show()