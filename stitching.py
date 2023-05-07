#%%
import cv2
import numpy as np
import operator
from PIL import Image
from matplotlib import pyplot as plt




#%%

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    
        
    #print(t)
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    dummy_result=np.zeros(result.shape)
    dummy_result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    
    

    overlap_1=cv2.warpPerspective(np.zeros(img2.shape)+1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    overlap_2=np.zeros(overlap_1.shape)
    overlap_2[t[1]:h1+t[1],t[0]:w1+t[0]] = np.zeros(img1.shape)+1
    overlap=overlap_1+overlap_2
    overlap2d=(overlap[:,:,0]+overlap[:,:,1]+overlap[:,:,2])==6
    
    #print(overlap_1.shape)
    #cv2.imshow("Matching Image",np.float32(overlap) )

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    newimg=np.zeros(overlap.shape)
    newimg=result/255
    newimg[t[1]:h1+t[1],t[0]:w1+t[0]] = img1/255
    for i in range(overlap2d.shape[0]):
        for j in range(overlap2d.shape[1]):
            if(overlap2d[i,j]==True):
                newimg[i,j]=(0.5*result[i,j]+0.5*dummy_result[i,j])/255
    #        for k in range(3):
    #            
    #                #print([i,j,k])
    #                #print(dummy_result[i,j,k])
    #                

    
    #cv2.imshow("Stitched Image", newimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return newimg






def stitch_and_blend(img1,img2):

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors for both images using SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Use a Brute-Force Matcher to match the keypoints
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    # Sort the matches by their distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw the top 10 matches on a new image
    #matching_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matching image
    #cv2.imshow("Matching Image", matching_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Use the matched keypoints to find the homography matrix and warp the second image onto the first image
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    #result[0:img1.shape[0], 0:img1.shape[1]] = img1
    #result[0:img2.shape[0], 0:img2.shape[1]] = img2
    #result = cv2.warpPerspective(img2, H, ((img1.shape[1]+img2.shape[1]), img1.shape[0]+100))
    result = warpTwoImages(img1, img2, H)
    # Display the stitched image
    #result.show()
    return result
#%%
# Load the images


img1 = cv2.imread("/mnt/c/Users/ryan7/Downloads/training-20230505T210618Z-001/training/input1/000003.jpg")
img2 = cv2.imread("/mnt/c/Users/ryan7/Downloads/training-20230505T210618Z-001/training/input2/000003.jpg")

#plt.imshow("Matching Image", result.astype('uint8'))
result=stitch_and_blend(img1,img2)
#img = Image.fromarray(result, 'RGB')
#img.show()
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
