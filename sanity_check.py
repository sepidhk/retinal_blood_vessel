# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np

masks_path= glob.glob("new_data/train/mask/*.png")

for mask_path in masks_path:
    mask_data= cv2.imread(mask_path, 0)
    mask_data = mask_data / 255.
    print(os.path.basename(mask_path), np.unique(mask_data))