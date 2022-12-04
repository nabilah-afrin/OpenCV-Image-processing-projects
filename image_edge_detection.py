import cv2

img = cv2.imread("C:\\Users\\Asus\\Downloads\\img.jpg")
#use the following line if you want to resize the image
resized = cv2.resize(img, (450, 600), interpolation = cv2.INTER_AREA)
edges = cv2.Canny(resized,100,100)
cv2.imshow('Original', resized)
cv2.imshow('Edges', edges)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
