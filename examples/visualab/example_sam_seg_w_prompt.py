"""
Image segmentation using SAM with prompt
============================================================

This example runs an interactive demo which allows user to select a region of interest and generate a mask for it.
"""

import os

import cv2
import matplotlib.pyplot as plt

import rofunc as rf

image_path = os.path.join(rf.oslab.get_rofunc_path(), "../examples/data/visualab/truck.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
rf.visualab.sam_predict(image,
                        use_point=True,
                        use_box=False,
                        choose_best_mask=True)
