#画像をQで閉じる

#https://qiita.com/ryo_ryo/items/973007667c528ef23abb
import numpy as np
import cv2

capture = cv2.VideoCapture(0)

class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, img):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [(x,y)]
            # cv2.circle(img, (x,y), 6, (0, 0, 255), 2) 
            # cv2.imshow('image', img)



def getPointsList(img):



    m = mouse_event_handler()

    cv2.imshow('img',img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback ("img", \
                          lambda event, x, y, flags, param: \
                          m.mouse_event(event, x, y, flags, param, img))

    while True:
        # cv2.imshow("img", dummy_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break

    cv2.destroyAllWindows()
    return m.points

dummy_img = 255*np.ones((1000,1000,3))



# カメラからフレームを1枚取得
ret, frame = capture.read()

if not ret:
    capture.release()
    cv2.destroyAllWindows()
    exit()


pts = np.array(getPointsList(frame))
print(pts)



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

while True:
    # カメラからのフレーム取得
    ret, frame = capture.read()
    if not ret:
        break

    # 射影変換
    # M = cv2.getPerspectiveTransform(p_original, p_trans)
    M = cv2.getPerspectiveTransform(rect, dst)
    frame_trans = cv2.warpPerspective(frame, M, (width, height))


    # 表示
    cv2.imshow('変換後の映像', frame_trans)
    # cv2.imshow('変換前の映像', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()