#%%
import cv2
import numpy as np
import operator
from PIL import Image
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join

label_folder = []
total_size = 0
data_path = "/mnt/c/Users/ryan7/Downloads/PanoramaStitchingP2-master/InputImages/Helens"



for root, dirts, files in os.walk(data_path):
    total_size += len(files)
    for dirt in dirts:
        label_folder.append(dirt)
        #total_size = total_size+  len(files)
print("found", total_size, "files.")
print("folder:", label_folder)

images=[]

for text in files:
    #text=text.replace(".jpg", "")

    images.append(cv2.imread(data_path+r'/'+text))
    


#%% for random order stitch with max match point match
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
stitcher.setPanoConfidenceThresh(0.0) 
for i in range(0,len(images)-1):
    number_of_last_max_match=0
    best_match_image=0
    already_stitch=[]
    skip=False
    for j in range(1+i,len(images)):
        for k in range(len(already_stitch)):
            if(already_stitch==[]):
                break
            if(j==already_stitch[k]):
                skip=True
        if (skip==True):
            continue
        gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY)

        # Initialize the KAZE detector and descriptor
        kaze = cv2.KAZE_create()
        kp1, des1 = kaze.detectAndCompute(gray1, None)
        kp2, des2 = kaze.detectAndCompute(gray2, None)

        # Use a brute-force matcher to match the keypoints
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        if(len(matches)>number_of_last_max_match):
            number_of_last_max_match=len(matches)
            best_match_image=j

    
    panorama = stitcher.stitch((images[0],images[best_match_image]))
    if panorama[1] is None:
        print(panorama[0])
        continue
    images[0]=panorama[1]
    already_stitch.append(best_match_image)

    cv2.imshow("Matching Image", panorama[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imshow("Matching Image", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("stitcher_multiple_random.jpg", images[0])
# %% in order stitch
images=[]

for i in range(len(files)):
    #text=text.replace(".jpg", "")

    images.append(cv2.imread(data_path+r'/'+files[i]))
#%% in order
panorama = images[0]
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
stitcher.setPanoConfidenceThresh(0.0) 

def stitch_with_neighbor(images):
    new_images=[]
    for i in range(1,round(len(images)/2)):
        if(round(len(images)/2)==i and len(images)%2!=0):
            new_images.append(images[i])
            continue
            
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        ret,panorama = stitcher.stitch((images[i+1],images[i]))
        if(ret!=0):
            new_images.append(images[i])
            new_images.append(images[i+1])
            continue
        cv2.imshow("Matching Image", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        new_images.append(panorama)
    return new_images

images=stitch_with_neighbor(images)
images=stitch_with_neighbor(images)
images=stitch_with_neighbor(images)

# %%
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#stitcher.setFeaturesFinder(cv2.KAZE_create())
panorama = stitcher.stitch(images)
cv2.imshow("Matching Image", panorama[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
import random
random.shuffle(images)
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
panorama = stitcher.stitch(images)
cv2.imshow("Matching Image", panorama[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
