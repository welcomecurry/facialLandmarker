import cv2
import dlib
from imutils import face_utils

# get webcam video
capture = cv2.VideoCapture(0)

# detect face coordinates
detector = dlib.get_frontal_face_detector()

# predict landmark keypoints on face from .dat file
predictor = dlib.shape_predictor('/home/amit/Desktop/facialLandmarkDetection/shape_predictor_68_face_landmarks.dat')

while True:
    # capture frame by frame and convert to grayscale
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # fetching face coordinates
    for face in faces:
        landmarks = predictor(gray, face)
        # get x and y coordinate of face points then draw dot
        # range can be modified based on dots wanted based on image facial_landmarks.jpg
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show frame
    cv2.imshow("Landmarker", frame)
    key = cv2.waitKey(1)

    # kills upon entering 'esc' or 'q'
    if(key == 27 or key == ord('q')):
        break