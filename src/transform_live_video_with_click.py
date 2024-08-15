# import the opencv library 
import cv2

class mouse_event_handler:
    def __init__(self):
        self.points = []

    def mouse_event(self, event, x, y, flags, param, frame):
        if event == cv2.EVENT_LBUTTONUP:
            self.points += [[x,y]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) 
            cv2.circle(frame, (x,y), 6, (0, 0, 255), 2) 


# define a video capture object 
vid = cv2.VideoCapture(0) 

def getList(vid):
    m = mouse_event_handler()

    while True:
        ret, frame = vid.read()
        cv2.setMouseCallback ("video", lambda event, x, y, flags, param: m.mouse_event(event, x, y, flags, param, frame))
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(m.points) >= 4:
            break
    vid.release() 
    cv2.destroyAllWindows()
    return m.points

print(getList(vid))
