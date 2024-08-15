#画像をQで閉じる

#https://qiita.com/ryo_ryo/items/973007667c528ef23abb
import numpy as np
import cv2
from transform import order_points
from getPointsList import getPointsList, getPointsListFromOneImage

capture = cv2.VideoCapture(0)

# カメラからフレームを1枚取得
ret, frame = capture.read()

if not ret:
    capture.release()
    cv2.destroyAllWindows()
    exit()

# 4点を取得
a = getPointsList(frame)
print("a:",a)
pts = np.array(a)


#ここから動画の表示


# フレームのサイズを取得
height, width, channels = frame.shape


rect = order_points(pts)
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
		[848, 0],
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