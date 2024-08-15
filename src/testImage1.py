
import cv2
import numpy as np
# from matplotlib import pyplot as plt


path = 'public/sample-desktop-1.png'
i = cv2.imread(path, 1) 

# 変換前後の対応点を設定
p_original = np.float32([[0,0], [473,55], [14, 514], [467, 449]])
p_trans = np.float32([[0,0], [524,0], [0,478], [524,478]])
 
# 変換マトリクスと射影変換
M = cv2.getPerspectiveTransform(p_original, p_trans)
i_trans = cv2.warpPerspective(i, M, (524, 478))


# # ここからグラフ設定
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
 
# # 画像をプロット
show = cv2.cvtColor(i_trans, cv2.COLOR_BGR2RGB)
# ax1.imshow(show)
 
# fig.tight_layout()
# plt.show()
# plt.close()
