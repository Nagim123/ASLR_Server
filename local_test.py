import cv2
from sign_detector import Recognizer

recognizer = Recognizer()

def webcum():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = recognizer.process_frame(img)
        cv2.imshow("CUM", img)
        cv2.waitKey(1)
webcum()