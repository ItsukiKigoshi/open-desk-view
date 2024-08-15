import cv2

from transform import four_point_transform
from testTransform2 import getPointsListFromOneImage

img = cv2.imread("src/sample-desktop-2.jpg")
cv2.imshow("img",img)
pts = getPointsListFromOneImage(img)

# the image
warped = four_point_transform(img, pts)
# show the original and warped images
cv2.imshow("Original", img)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows