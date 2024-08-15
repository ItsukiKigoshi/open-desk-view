import cv2
import numpy as np

# 画像のパスを指定
path = 'public/sample-desktop-1.jpg'
i = cv2.imread(path, 1) 

# 変換前後の対応点を設定
p_original = np.float32([[0, 0], [473, 55], [14, 514], [467, 449]])
p_trans = np.float32([[0, 0], [524, 0], [0, 478], [524, 478]])

# 変換マトリクスと射影変換
M = cv2.getPerspectiveTransform(p_original, p_trans)
i_trans = cv2.warpPerspective(i, M, (524, 478))

# 変換後の画像を表示
cv2.imshow('Transformed Image', i_trans)

# 'q'キーが押されたら終了
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
