import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('cat.png')
img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img3 = cv2.GaussianBlur(img2, (111, 111), 0)
image_gray=cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE) 
image_gray= cv2.GaussianBlur(image_gray,(11,11),0)  

img4 = cv2.Canny(image_gray, 50, 150)

images = [img1, img2, img3, img4]
titles = ["Original", "Gray", "Blur", "Edges"]

rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

plt.show()