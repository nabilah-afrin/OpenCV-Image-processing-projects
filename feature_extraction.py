import cv2

img1 = cv2.imread("C:\\Users\\Asus\\Downloads\\alchemist.jpg")
img1 = cv2.resize(img1, (450, 450), interpolation = cv2.INTER_AREA)
img2 = cv2.imread("C:\\Users\\Asus\\Downloads\\alchemist_full.jpg")
img2 = cv2.resize(img2, (600, 450), interpolation = cv2.INTER_AREA)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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
cv2.destroyAllWindows()



