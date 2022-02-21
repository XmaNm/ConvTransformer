import numpy as np
import cv2
import os
file = "ROI/"
X99 = []
X00 = []
X01 = []
for root, dirs, files in os.walk(file):
    for f in files:
        fs = os.path.join(root, f)
        # print(fs)
        img = cv2.imread(fs)
        if "1999" in fs:
            img = cv2.imread(fs)
            X99.append(img)
        if "2000" in fs:
            img = cv2.imread(fs)
            X00.append(img)
        if "2001" in fs:
            img = cv2.imread(fs)
            X01.append(img)
