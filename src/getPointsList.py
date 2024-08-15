import numpy as np
import cv2



class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, img):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [(x,y)]
            print(x,y)
            cv2.circle(img, (x,y), 6, (0, 0, 255), 2) 
            cv2.imshow('image', img)


def getPointsList(img):
    capture = cv2.VideoCapture(0)
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

    cv2.destroyWindow('img')
    return m.points




class mouse_event_handler_image:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, img):
        points_x = x
        points_y = y
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [(points_x,points_y)]
            print(points_x,points_y)
            cv2.circle(img, (points_x,points_y), 6, (0, 0, 255), 2) 
            cv2.imshow('img', img)


def getPointsListFromOneImage(img):

    m = mouse_event_handler_image()

    cv2.imshow('img',img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback ("img", \
                          lambda event, x, y, flags, param: \
                          m.mouse_event(event, x, y, flags, param, img))

    while True:

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break

    cv2.destroyAllWindows()
    return m.points
