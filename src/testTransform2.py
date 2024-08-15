#画像をQで閉じる

#https://qiita.com/ryo_ryo/items/973007667c528ef23abb
import numpy as np
import cv2
from transform import order_points

capture = cv2.VideoCapture(0)

class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, img):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [(x,y)]
            print(x,y)


def getPointsList(img):

    m = mouse_event_handler()

    cv2.imshow('img',img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback ("img", \
                          lambda event, x, y, flags, param: \
                          m.mouse_event(event, x, y, flags, param, img))

    while True:
        # カメラからのフレーム取得
        ret, frame = capture.read()
        if not ret:
            continue

        # 表示
        cv2.imshow('img', frame)

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break

    cv2.destroyAllWindows()
    return m.points





# カメラからフレームを1枚取得
ret, frame = capture.read()

if not ret:
    capture.release()
    cv2.destroyAllWindows()
    exit()

a = getPointsList(frame)
print("a:",a)
pts = np.array(a)


#ここから動画の表示
# フレームのサイズを取得
height, width, channels = frame.shape


# def order_points(pts):
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect



def order_points_new(pts):
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




print("pts:",pts)
rect = order_points_new(pts)
(tl, tr, br, bl) = rect
print("rect:",rect)
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
		[0, 0],
		[848 - 1, 0],
		[848 - 1, 592 - 1],
		[0, 592 - 1]], dtype = "float32")


if(len(pts) == 4):
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
        cv2.imshow('transformed', frame_trans)
        # cv2.imshow('変換前の映像', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()