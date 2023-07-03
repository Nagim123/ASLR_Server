import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
from sign_detector import Recognizer

recognizer = Recognizer()

def webcum():
    cap = cv2.VideoCapture(0)
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = recognizer.process_frame(img)
        cv2.imshow("CUM", img)
        cv2.waitKey(1)
webcum()