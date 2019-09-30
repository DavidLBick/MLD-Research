from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import psutil

for _, _, files in os.walk("./plots"):
  for file_i in files:
    img = mpimg.imread("./plots/" + file_i)
    plt.imshow(img)
    plt.show()
