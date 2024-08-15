import cv2
import numpy as np
import math

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

# 変換前後の対応点を設定
a = width / 2
b = a * X / math.sqrt(X**2 + Y**2)

A = [width/2, height - Y]
B = [width, height - Y]
C = [width, height]
D = [width/2, height]

# A_trans = [width/2, height - Y]
# B_trans = [width, height - Y]
# C_trans = [width, height]
# D_trans = [width/2, height]


Y = a
print(f'Y:{Y}')


A_trans = [width/2,0]
B_trans = [width/2 + a,0]
C_trans = [width/2 + a,Y]
D_trans = [width/2,Y]
print(A_trans)


# 変換前後の対応点
p_original = np.float32([A, B, C, D])
p_trans = np.float32([A_trans, B_trans, C_trans, D_trans])

while True:
    # カメラからのフレーム取得
    ret, frame = capture.read()
    if not ret:
        break

    # 射影変換
    M = cv2.getPerspectiveTransform(p_original, p_trans)
    frame_trans = cv2.warpPerspective(frame, M, (width, height))

    # A_trans、B_transの各点に円を描画する
    cv2.circle(frame_trans, (int(A_trans[0]), int(A_trans[1])), 5, (255, 0, 0), -1)
    cv2.circle(frame_trans, (int(B_trans[0]), int(B_trans[1])), 5, (255, 0, 0), -1)

    # 表示
    cv2.imshow('変換後の映像', frame_trans)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()