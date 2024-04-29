"""
Image segmentation using SAM
============================================================

This example generates all masks automatically with SAM.
"""

import os

import cv2
import matplotlib.pyplot as plt

import rofunc as rf

image_path = os.path.join(rf.oslab.get_rofunc_path(), "../examples/data/visualab/dog.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
rf.visualab.sam_generate(image)
