import cv2
import numpy as np
import argparse

# カメラの高さと目標の座標を設定
camera_height = 100
Y = 200
X = camera_height

# カメラキャプチャのオブジェクトを作成
capture = cv2.VideoCapture(0)

# カメラからフレームを1枚取得
ret, frame = capture.read()

if not ret:
    capture.release()
    cv2.destroyAllWindows()
    exit()

# フレームのサイズを取得
height, width, channels = frame.shape


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
pts = np.array(eval(args["coords"]), dtype = "float32")

rect = order_points(pts)
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
		[maxWidth/4, maxHeight/4],
		[maxWidth*3/4 , maxHeight/4],
		[maxWidth*3/4 , maxHeight*3/4 ],
		[maxWidth/4, maxHeight*3/4 ]], dtype = "float32")

def onMouse(event, x, y, corners):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        corners.add([x,y])



a = []

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
            a.append([x, y])
            print(a)
            # displaying the coordinates 
            # on the image window 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, str(x) + ',' +
                        str(y), (x,y), font, 
                        1, (255, 0, 0), 2) 
            cv2.circle(img, (x,y), 6, (0, 0, 255), 2) 
            cv2.imshow('image', img)
            

# driver function 
if __name__=="__main__": 
  
    # reading the image 
    ret, frame = capture.read()

    # 射影変換
    # M = cv2.getPerspectiveTransform(p_original, p_trans)
    M = cv2.getPerspectiveTransform(rect, dst)
    img = cv2.warpPerspective(frame, M, (width, height))
  
    # displaying the image 
    cv2.imshow('変換後の映像', img)

    corners = []
    cv2.setMouseCallback('image',click_event) 

    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 

	# close the window 
    cv2.destroyAllWindows() 

    # while True:
    #     if(len(corners)==4):
    #         break
  
    # while True:
    #     # カメラからのフレーム取得
    #     ret, frame = capture.read()
    #     if not ret:
    #         break

    #     # 射影変換
    #     # M = cv2.getPerspectiveTransform(p_original, p_trans)
    #     M = cv2.getPerspectiveTransform(rect, dst)
    #     frame_trans = cv2.warpPerspective(frame, M, (width, height))
        


    #     # 表示
    #     cv2.imshow('変換後の映像', frame_trans)
    #     cv2.setMouseCallback('image',onMouse) 

    #     # cv2.imshow('変換前の映像', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break


# while True:
#     # カメラからのフレーム取得
#     ret, frame = capture.read()
#     if not ret:
#         break

#     # 射影変換
#     # M = cv2.getPerspectiveTransform(p_original, p_trans)
#     M = cv2.getPerspectiveTransform(rect, dst)
#     frame_trans = cv2.warpPerspective(frame, M, (width, height))


#     # 表示
#     cv2.imshow('変換後の映像', frame_trans)
#     # cv2.imshow('変換前の映像', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

capture.release()
cv2.destroyAllWindows()