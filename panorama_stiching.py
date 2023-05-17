#%%
import cv2
import numpy as np
import operator
from PIL import Image
from matplotlib import pyplot as plt


#%%
img1 = cv2.imread("/mnt/c/Users/ryan7/Downloads/training-20230505T210618Z-001/training/input1/000001.jpg")
img2 = cv2.imread("/mnt/c/Users/ryan7/Downloads/training-20230505T210618Z-001/training/input2/000001.jpg")
#cv2.imshow("Stitched Image", result)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

result = stitcher.stitch((img1,img2))

cv2.imwrite("stitcher_result.jpg", result[1])
# %%
