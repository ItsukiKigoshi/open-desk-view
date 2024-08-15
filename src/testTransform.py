from transform import four_point_transform
import cv2
import numpy as np



# カメラキャプチャのオブジェクトを作成
capture = cv2.VideoCapture(0)

# カメラからフレームを1枚取得
ret, frame = capture.read()

if not ret:
    capture.release()
    cv2.destroyAllWindows()
    exit()

a=[]
print(a)

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
            if len(a) == 4:
                pts = np.array(a)
                # print(pts)
                cv2.waitKey(1) 

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

    while True:
        if cv2.waitKey(1):
            cv2.waitKey(0)
            break
      
    # while True:
    #     # カメラからのフレーム取得
    #     ret, frame = capture.read()
    #     if not ret:
    #         break

    #     # 射影変換
    #     # M = cv2.getPerspectiveTransform(p_original, p_trans)
    #     # M = cv2.getPerspectiveTransform(rect, dst)
    #     # frame_trans = cv2.warpPerspective(frame, M, (width, height))
        


    #     # 表示
    #     cv2.imshow('変換後の映像', frame)

    #     # cv2.imshow('変換前の映像', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
