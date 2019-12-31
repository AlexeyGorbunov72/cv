import cv2, time
from imutils.video import VideoStream, FPS

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
while True:
    print(vs.read())
    cv2.imshow("sa", vs.read())
    cv2.waitKey(0)
