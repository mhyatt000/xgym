import cv2
img = cv2.imread("test_frame.png")
print(img.shape)
cv2.imshow("test", img)
cv2.waitKey(0)

