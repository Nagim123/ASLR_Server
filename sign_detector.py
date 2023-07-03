import cv2
import numpy as np
import torch
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
from s_model import SequenceModel
import torch.nn.functional as F

FRAMES_PER_PREDICTION = 30#308
PREDICT_THRESHOLD = 0.5
SIGN_CLASSES = ["HELLO", "I LOVE YOU", "THANKS"]

class Recognizer:
    def __init__(self) -> None:
        self._hands_detector = HandDetector()
        self._pose_detector = PoseDetector()
        self._sequence = []
        self.model = SequenceModel()
        self.model.load_state_dict(torch.load("model/best.pt"))
        self.model.eval()
        self.current_prediction = "Unknown"

    def process_frame(self, frame: np.array) -> np.array:
        img = frame
        self._collect_points(img)
        img = self._predict(img)
        return img
    
    def _collect_points(self, frame: np.array):
        points = [0] * (21 * 3 * 2 + 33 * 3)
        # Recognize hands and collect them into list of all points
        hands, img1 = self._hands_detector.findHands(frame)
        for i in range(len(hands)):
            ind_shift = 0
            if hands[i].get('type') == 'Left':
                ind_shift = 21 * 3
            hand_points = hands[i].get('lmList')
            for j in range(len(hand_points)):
                for k in range(3):
                    points[ind_shift + j * 3 + k] = hand_points[j][k]

        # Recognize the pose and collect points
        img2 = self._pose_detector.findPose(frame)
        lmList, bboxInfo = self._pose_detector.findPosition(frame, bboxWithHands=False)
        for i in range(len(lmList)):
            for j in range(1, 4):
                points[21 * 3 * 2 + i * 3 + j - 1] = lmList[i][j]
        
        self._sequence.append(points)
        self._sequence = self._sequence[-FRAMES_PER_PREDICTION:]

    def _predict(self, frame:np.array):
        if len(self._sequence) == FRAMES_PER_PREDICTION:
            with torch.no_grad():
                t_seq = torch.tensor([self._sequence])
                t_seq = F.normalize(t_seq.float(), dim=1)
                prediction = self.model(t_seq)
                class_id = np.argmax(prediction)
                self.current_prediction = SIGN_CLASSES[class_id] + "|" + str(prediction[0][class_id])
        
        img = cv2.putText(frame, self.current_prediction, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2,cv2.LINE_AA)
        return img


