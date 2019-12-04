import opendr
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../demo_wild/input/Duncan.jpg',)
print(img.shape)
plt.imshow(img,'gray')
plt.show()
cv.imshow('test',img)
cv.waitKey(0)