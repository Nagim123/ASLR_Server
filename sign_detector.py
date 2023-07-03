import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector


FRAMES_PER_PREDICTION = 308
PREDICT_THRESHOLD = 0.5

class Recognizer:
    def __init__(self) -> None:
        self._hands_detector = HandDetector()
        self._pose_detector = PoseDetector()
        self._sequence = []

    def process_frame(self, frame: np.array) -> np.array:
        hands, img = self._hands_detector.findHands(frame)
        img = self._pose_detector.findPose(img)
        # lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=False)
        
        return img
    
    def _convert_to_point_array(self):
        pass

    def _predict(self):
        if len(self._sequence) == 308:
            pass


