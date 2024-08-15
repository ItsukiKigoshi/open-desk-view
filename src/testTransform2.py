#画像をQで閉じる
import numpy as np
import cv2

class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, img):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [[x,y]]
            # line_number, buf = divmod(len(self.points),2)
            # is_even = (buf == 0)
            # if is_even:
            #     p1 = self.points[(line_number-1)*2]
            #     p2 = self.points[(line_number-1)*2+1]
            #     cv2.line(img,p1,p2,(255,0,0),3)


def getList(img):
    m = mouse_event_handler()

    cv2.imshow('img',img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback ("img", \
                          lambda event, x, y, flags, param: \
                          m.mouse_event(event, x, y, flags, param, img))

    while (True):
        # cv2.imshow("img", dummy_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break

    cv2.destroyAllWindows()
    return m.points

dummy_img = 255*np.ones((1000,1000,3))
print(getList(dummy_img))

