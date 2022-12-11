


import cv2
 
def sobel_edge_detection(img):
    # Sobel operator looks in sudden change in pixel intensity
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection lookes for edges enhanced in X-direction
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection lookes for edges enhanced in X-direction
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)
    return sobel_edge_detection
 
def Canny_Edge_Detection(img_blur):
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

# Read the original image
img = cv2.imread("E:\AJAIRA\Animu\FHYsJyuVcAYsrmp.jpg") 
cv2.imshow('Original', img)# Display original image
cv2.waitKey(0)
# Convert to graycsale as edge doesnt require color
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# gaussian blur reduces the noise
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) #3x3 = kernel size with degree of blurring

sobel_img = sobel_edge_detection(img_blur)
Canny_Edge_Detection(img_blur)

cv2.destroyAllWindows()


