"""
Image segmentation using EfficientSAM with prompt
============================================================

This example runs an interactive demo which allows user to select a region of interest and generate a mask for it.
It can be used on edge devices like Nvidia Jetson Nano/TX2/Xavier NX/AGX with a higher speed than SAM.
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
rf.visualab.efficient_sam_predict(image,
                                  use_point=False,
                                  use_box=True,
                                  efficient_sam_checkpoint="efficientsam_s_gpu.jit")
