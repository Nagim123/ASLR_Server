import cv2
import numpy as np
import torch
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
from s_model import GruModel
import torch.nn.functional as F

FRAMES_PER_PREDICTION = 30#308
PREDICT_THRESHOLD = 0.9
SIGN_CLASSES = ["ATTENTION", "EMPTY", "FOR", "HELLO", "HUMAN", "LIKE", "PROGRAMMING", "THANK YOU", "WE"]
MAX_SENTENCE_SIZE = 6

# To solve problem with rapidly changing predictions, just collect a lot of predictions and get the class that
# presented in most of them.

class Recognizer:
    def __init__(self) -> None:
        self._hands_detector = HandDetector()
        self._pose_detector = PoseDetector()
        self._sequence = []
        self._predictions = []
        self._sentence = []
        self.model = GruModel(177, 128, 2, 9)
        self.model.load_state_dict(torch.load("model/best_gru_128_2_0927.pt", map_location=torch.device("cpu")))
        self.model.eval()
        self.current_prediction = "Unknown"

    def process_frame(self, frame: np.array) -> np.array:
        img = frame
        self._collect_points(img)
        img = self._predict(img)
        return img
    
    def _collect_points(self, frame: np.array):
        points = [0] * (21 * 3 * 2 + 17 * 3)
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
        lmList = lmList[:17]
        for i in range(len(lmList)):
            for j in range(1, 4):
                points[21 * 3 * 2 + i * 3 + j - 1] = lmList[i][j]
        
        self._sequence.append(points)
        self._sequence = self._sequence[-FRAMES_PER_PREDICTION:]

    def _predict(self, frame:np.array):
        if len(self._sequence) == FRAMES_PER_PREDICTION:
            with torch.no_grad():
                t_seq = torch.tensor([self._sequence])
                t_seq = F.normalize(t_seq.float())
                prediction = self.model(t_seq)
                class_id = np.argmax(prediction)
                if prediction[0][class_id] > PREDICT_THRESHOLD:
                    self._predictions.append(class_id.item())
                    self._predictions = self._predictions[-10:]
                    #if np.all(np.array(self._predictions) == class_id.item()):
                    self.current_prediction = SIGN_CLASSES[class_id] + "|" + str(prediction[0][class_id])
                    if len(self._sentence) == 0 or self._sentence[-1] != SIGN_CLASSES[class_id]:
                        self._sentence.append(SIGN_CLASSES[class_id])
                        self._sentence = self._sentence[-MAX_SENTENCE_SIZE:]
        img = cv2.putText(frame, self.current_prediction, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2,cv2.LINE_AA)
        return img

    def get_current_sentence(self) -> str:
        return " ".join(self._sentence)
    
    # Custom point array normalization
    def _pivot_normalization(self, tensor_sequence):
        tensor_sequence = tensor_sequence[0]
        normalized_sequence = []
        
        for tensor in tensor_sequence:
            normalized_tensor = []
            x_values = []
            y_values = []
            z_values = []
            k = 0
            list_data = tensor.tolist()
            for data in list_data:
                if k == 0:
                    x_values.append(data)
                elif k == 1:
                    y_values.append(data)
                elif k == 2:
                    z_values.append(data)
                    k = -1
                k += 1
            pivot_point_index = 42
            max_x_val = max(x_values)
            max_y_val = max(y_values)
            max_z_val = max(z_values)
            for i in range(0, len(x_values)):
                normalized_tensor.append((x_values[i] - x_values[pivot_point_index])/(max_x_val+1))
                normalized_tensor.append((y_values[i] - y_values[pivot_point_index])/max_y_val)
                normalized_tensor.append((z_values[i] - z_values[pivot_point_index])/max_z_val)
            normalized_sequence.append(normalized_tensor)

        return torch.tensor([normalized_sequence])