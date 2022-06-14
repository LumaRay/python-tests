#!/usr/bin/env python

import os

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

test_images = [f for f in os.listdir('test') if f.endswith('.jpg')]
import random
img_path = "test/" + "0" # random.choice(test_images)
imShow('predictions.jpg')