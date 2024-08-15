
import cv2
import numpy as np
# from matplotlib import pyplot as plt


path = 'public/sample-desktop-1.jpg'
i = cv2.imread(path, 1)

# フレームのサイズを取得
height, width, channels = i.shape




A = [1312,1000]
B = [2837,1000]
C = [3925,2239]
D = [529,2239]


a = abs(C[0] - D[0])
b = abs(A[0] - B[0])
c = abs(A[1] - D[1])
print(f'a:{a}')
print(f'b:{b}')
print(f'c:{c}')

Y = a
print(f'Y:{Y}')


A_trans = [width/2,0]
B_trans = [width/2 + a,0]
C_trans = [width/2 + a,Y]
D_trans = [width/2,Y]
print(A_trans)


# 変換前後の対応点を設定
p_original = np.float32([A, B, C, D])



p_trans = np.float32([A_trans, B_trans, C_trans, D_trans])
 
# 変換マトリクスと射影変換
M = cv2.getPerspectiveTransform(p_original, p_trans)
i_trans = cv2.warpPerspective(i, M, (width*2, height*2))


# ここからグラフ設定
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
 
# 画像をプロット
show = cv2.cvtColor(i_trans, cv2.COLOR_BGR2RGB)

cv2.imshow("tekitou",show)
 
# fig.tight_layout()
# plt.show()
# plt.close()