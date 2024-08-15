# import the opencv library 
import cv2
import numpy as np
from transform import four_point_transform

class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, frame):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [[x,y]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) 
            cv2.circle(frame, (x,y), 6, (0, 0, 255), 2)


# define a video capture object 
vid = cv2.VideoCapture(0) 

def getList(vid):
    m = mouse_event_handler()
    while True:
        ret, frame = vid.read()
        cv2.setMouseCallback ("video", lambda event, x, y, flags, param: m.mouse_event(event, x, y, flags, param, frame))
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break
    return np.array(m.points)

pts = getList(vid)
print(pts)

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

rect = order_points(pts)
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0, 0],
    [848, 0],
    [848 - 1, 592 - 1],
    [0, 592 - 1]], dtype = "float32")

while True:
    # カメラからのフレーム取得
    ret, frame = vid.read()

    # 射影変換
    # M = cv2.getPerspectiveTransform(p_original, p_trans)
    M = cv2.getPerspectiveTransform(rect, dst)
    frame_trans = cv2.warpPerspective(frame, M,  (842, 595))
    frame_flip = cv2.flip(frame_trans, -1)

    # 表示
    cv2.imshow('Transformed Frame', frame_flip)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()