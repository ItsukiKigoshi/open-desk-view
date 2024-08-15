from transform import four_point_transform
import cv2
import numpy as np

a=[]
print(a)

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
            a.append([x, y])
            # displaying the coordinates 
            # on the image window 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, str(x) + ',' +
                        str(y), (x,y), font, 
                        1, (255, 0, 0), 2) 
            cv2.circle(img, (x,y), 6, (0, 0, 255), 2) 
            cv2.imshow('image', img)
            if len(a) == 4:
                pts = np.array(a)
                print(pts)
                cv2.destroyAllWindows() 

# driver function 
if __name__=="__main__": 

	# reading the image 
	img = cv2.imread('public/sample-desktop-1.jpg', 1) 

	# displaying the image 
	cv2.imshow('image', img) 

	# setting mouse handler for the image 
	# and calling the click_event() function 
	cv2.setMouseCallback('image', click_event) 

	# wait for a key to be pressed to exit 
	cv2.waitKey(0) 

	# close the window 
	cv2.destroyAllWindows() 
