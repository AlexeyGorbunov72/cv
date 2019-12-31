# import the necessary packages
import imutils
from imutils import face_utils
from imutils.video import VideoStream
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
vs = VideoStream(0).start()
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image and convert it to grayscale
while True:
    frame = vs.read()
    print(frame)
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    for rect in rects:
        # determine the facial landmarks for the face region, then

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        # detect faces in the grayscale frame
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart : mEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouth_ = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (128, 128, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (205, 0, 205), 1)
        cv2.drawContours(frame, [mouth_], -1, (255, 255, 0), 1)
        # 2F4F4F
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
