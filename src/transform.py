# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

import numpy as np
import cv2
# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	print("rect:",rect)
# 	return rect


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    # x の値でソート
    sorted_indices = np.argsort(pts[:, 0])
    sorted_data = pts[sorted_indices]

    # リストを前後2つに分割
    first_half = sorted_data[:2]
    second_half = sorted_data[2:]

    # y の値でソート
    first_half_sorted = first_half[np.argsort(first_half[:, 1])]
    second_half_sorted = second_half[np.argsort(second_half[:, 1])]

    rect[0] = first_half_sorted[0]
    rect[3] = first_half_sorted[1]

    rect[1] = second_half_sorted[0]
    rect[2] = second_half_sorted[1]

    return rect




def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	# dst = np.array([
	# 	[0, 0],
	# 	[maxWidth - 1, 0],
	# 	[maxWidth - 1, maxHeight - 1],
	# 	[0, maxHeight - 1]], dtype = "float32")
	dst = np.array([
		[0, 0],
		[848, 0],
		[848 - 1, 592 - 1],
		[0, 592 - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	print(M)
	# warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	warped = cv2.warpPerspective(image, M, (842, 595))
	# return the warped image
	return warped