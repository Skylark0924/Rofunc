"""
Part-level segmentation using SAM and VLPart with prompt
============================================================

This example allows user to provide a text prompt and generate a mask for the corresponding object part.
"""

import os

import cv2

import rofunc as rf

# obtain rgb and depth image of the grasping scene
image_path = os.path.join(rf.oslab.get_rofunc_path(), "../examples/data/visualab/knife2.png")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

text_prompt = "knife handle"
affordance_masks = rf.visualab.vlpart_sam_predict(image, text_prompt)
