# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# Usage: "% python src/transform_example.py --image public/sample-desktop-1.jpg --coords "[(275, 329), (785,321), (914,584), (141,592)]"           "

# import the necessary packages

#[(275, 329), (785,321), (914,584), (141,592)]

# python src/transform_example.py --image public/sample-desktop-1.jpg --coords "[(281, 306), (802, 602), (972, 416), (953, 323)]"

from transform import four_point_transform
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)