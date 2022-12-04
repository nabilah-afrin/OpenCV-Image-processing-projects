import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Asus\\Downloads\\Capture.JPG")
img_rot = cv2.imread("C:\\Users\\Asus\\Downloads\\Capture_180.JPG")

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# match descriptors of both images
matches = bf.match(descriptors_1,descriptors_2)

# sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
# draw first 50 matches
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
print("Matches Found :" + str(len(matches)))
# show the image
cv2.imshow('image', matched_img)
# save the image
cv2.imwrite("matched_images.jpg", matched_img)
cv2.waitKey(0)

#################################  ORB  ######################################
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Asus\\Downloads\\Capture.JPG")
img_rot = cv2.imread("C:\\Users\\Asus\\Downloads\\Capture_180.JPG")

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000)

keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# match descriptors of both images
matches = bf.match(descriptors_1,descriptors_2)

# sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
# draw first 50 matches
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
print("Matches Found :" + str(len(matches)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray_32 = np.float32(gray)
#haris = cv2.cornerHarris(gray_32, 2, 3, 0.04)
#img[haris>0.01*haris.max()] = [0, 0, 255]

corners = cv2.goodFeaturesToTrack(gray,150,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img2,(x,y),3,255,-1)

#cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)  # waits until a key is pressed

orb = cv2.ORB_create(500)
kp, des = orb.detectAndCompute(img, None)

img2 = cv2.drawKeypoints(img, kp, None, flags=0)

cv2.imshow('img2', img2)
cv2.waitKey(0)

sift = cv2.SIFT_create()
kp = sift.detect(img1, None)

img2 = cv2.drawKeypoints(img1, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img2', img2)
cv2.waitKey(0)