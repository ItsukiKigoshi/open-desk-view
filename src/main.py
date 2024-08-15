import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    canny_frame = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 10, 100)
    cv2.imshow('frame', canny_frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()