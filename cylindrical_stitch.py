#%%
import os
import cv2
import math
import numpy as np
      

def ReadImage(ImageFolderPath):
    Images = []									# Input Images will be stored in this list.

	# Checking if path is of folder.
    if os.path.isdir(ImageFolderPath):                              # If path is of a folder contaning images.
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)):                     # Getting all image's name present inside the folder.
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)  # Reading images one by one.
            
            # Checking if image is read
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage)                               # Storing images.
            
    else:                                       # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

    
def FindMatches(BaseImage, SecImage):
    # Using SIFT to find the keypoints and decriptors in the images
    #Sift = cv2.SIFT_create()
    #BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    #SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)


    kaze = cv2.AKAZE_create()
    BaseImage_kp, BaseImage_des = kaze.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = kaze.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

        # Use a brute-force matcher to match the keypoints
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    #matches = bf.match(BaseImage_des, SecImage_des)

        # Sort the matches by distance
    #matches = sorted(matches, key=lambda x: x.distance)



    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applytng ratio test and filtering out the good matches.
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

    return GoodMatches, BaseImage_kp, SecImage_kp



def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, exit the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status

    
def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xt, yt) is the coordinate of the i th corner of the image. 
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0] #correction is useful when stitch left image
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely 
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix



def StitchImages(BaseImage, SecImage):
    # Applying Cylindrical projection on SecImage
    SecImage_Cyl, mask_x, mask_y = ProjectOntoCylinder(SecImage)
    #cv2.imshow("Matching Image", SecImage_Cyl)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    
    # Getting SecImage Mask
    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255

    # Finding matches between the 2 images and their keypoints
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage_Cyl)
    
    # Finding homography matrix.
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    # Finding size of new frame of stitched images and updating the homography matrix 
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage_Cyl.shape[:2], BaseImage.shape[:2])

    # Finally placing the images upon one another.
    SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

    return StitchedImage


def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    #f = 1100       # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens
    
    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    #print(InitialImage.shape)
    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]
    
    # Finding corresponding coordinates of the transformed image in the initial image
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    # Finding transformed image points whose corresponding 
    # initial image points lies inside the initial image
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    # Removing all the outside points from everywhere
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]
    
    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    # Bilinear interpolation
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    
    TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                      ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                      ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                      ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)
    #print(min_x)
    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x : -min_x, :]
    #print(TransformedImage.shape)
    return TransformedImage, ti_x-min_x, ti_y

def swap0(s1, s2):
    #assert type(s1) == list and type(s2) == list
    tmp = s1
    s1 = s2
    s2 = tmp
    return s1,s2

def find_focal(H):
    f1=0
    f0=0
    H=np.array(np.float32(H))
    H=H.flatten()
    v1= H[0]*H[3]+H[1]*H[4]
    v2=H[0]**2+H[1]**2-H[3]**2-H[4]**2
    c1=(H[5]**2-H[2]**2)/v1
    c2=(-1*H[2]*H[5])/v2
    f1_ok=True
    if(c1<c2):
        c1,c2=swap0(c1,c2)
        
    if(c1>0 and c2>0):

        if(abs(v1)>abs(v2)):
            f0= math.sqrt(max(c1,c2))
                

        else:
            f0= math.sqrt(min(c1,c2))
    elif(c1>0):
        f0=math.sqrt(c1)
    else :
        f1_ok=False
    
    v1=H[7]*H[6]
    v2=(H[7]-H[6])*(H[7]+H[6])
    c1=(-1*(H[0]*H[1]+H[3]*H[4]))/v2
    c2=(H[0]**2+H[3]**2-H[1]**2-H[4]**2)/v1
    f2_ok=True
    if(c1<c2):
        c1,c2=swap0(c1,c2)
    if(c1>0 and c2>0):
        if(abs(v1)>abs(v2)):
            f1=math.sqrt(max(c1,c2))
        else:
            f1=math.sqrt(min(c1,c2))
    elif(c1>0):
        f1=math.sqrt(c1)
    else :
        f2_ok=False
    return f0,f1,f1_ok,f2_ok

def set_focal():
    global f 

    focal_list=[]   #estimate focal
    for i in range(0, len(Images)):
        for j in range(i+1,len(Images)):

        #kaze = cv2.KAZE_create()
        #BaseImage_kp, BaseImage_des = kaze.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
        #SecImage_kp, SecImage_des = kaze.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)
        # Finding matches between the 2 images and their keypoints
            Matches, BaseImage_kp, SecImage_kp = FindMatches(Images[i], Images[j])
            if(len(Matches)<4):
                continue
        # Finding homography matrix.
            HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
            #print(Status)
            if(HomographyMatrix is None):
                continue
            f0,f1,f1_ok,f2_ok=find_focal(HomographyMatrix)
            if(f1_ok and f2_ok):
                focal_list.append(math.sqrt(f1*f0))
            #focal_list.append(f1)
    import statistics
    f=sum(focal_list)/len(focal_list)
    print(f)
    return f
    


#%% stitch in randonm
#if __name__ == "__main__":
    # Reading images.
Images = ReadImage("/mnt/c/Users/ryan7/Downloads/PanoramaStitchingP2-master/InputImages/Suns")
import random
random.shuffle(Images)
BaseImage, _, _ = ProjectOntoCylinder(Images[0])

already_stitch=[0,]
for i in range(1, len(Images)):

    
    number_of_last_max_match=0 #find the most matches point image and then stitched them
    best_match_image=0
    for j in [j for j in range(1,len(Images)) if j not in already_stitch]:
        #print(j)
        #print(j not in already_stitch)
        #skip=False
        #for k in range(len(already_stitch)):
        #    if(already_stitch==[]):
         #       break
        #    if(j==already_stitch[k]):
        #        skip=True
        #if (skip==True):
        #    continue
        gray1 = cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(Images[j], cv2.COLOR_BGR2GRAY)

        # Initialize the KAZE detector and descriptor
        kaze = cv2.AKAZE_create()
        kp1, des1 = kaze.detectAndCompute(gray1, None)
        kp2, des2 = kaze.detectAndCompute(gray2, None)

        # Use a brute-force matcher to match the keypoints
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        print(len(matches))
        # Sort the matches by distance
        #matches = sorted(matches, key=lambda x: x.distance)
        if(len(matches)>number_of_last_max_match):
            number_of_last_max_match=len(matches)
            best_match_image=j
    StitchedImage = StitchImages(BaseImage, Images[best_match_image])
    already_stitch.append(best_match_image)
    #print(best_match_image)
    #print(already_stitch)

    BaseImage = StitchedImage.copy()
    cv2.imshow("Matching Image", BaseImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

cv2.imwrite("Stitched_Panorama_random.png", BaseImage)


#%% stitch in order and with estimated f
Images = ReadImage("/mnt/c/Users/ryan7/Downloads/PanoramaStitchingP2-master/InputImages/Rainier")
#%%
sumpixel=0
for i in range(0, len(Images)):
    sumpixel+=Images[i].shape[0]*Images[i].shape[1]
workscale=min(1,math.sqrt(0.6*10**6/sumpixel))
for i in range(0, len(Images)):
    Images[i]=cv2.resize(Images[i],(round(Images[i].shape[1]*workscale),round(Images[i].shape[0]*workscale)))
    #cv2.imshow("Matching Image",Images[i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#%%
global f
f=set_focal() #f=1108

#%%

#f=1100
BaseImage, _, _ = ProjectOntoCylinder(Images[0])


for i in range(1, len(Images)):
    StitchedImage = StitchImages(BaseImage, Images[i])

    BaseImage = StitchedImage.copy()
    #cv2.imshow("Matching Image", BaseImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()     

cv2.imwrite("Stitched_Panorama_Field.png", BaseImage)

# %%

        
        
