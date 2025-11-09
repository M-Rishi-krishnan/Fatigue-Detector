# real_time_predictor.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import json
from collections import deque


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

class FeatureExtractor:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.landmark_indices = self.config['landmark_indices']

    def _calculate_aspect_ratio(self, landmarks, indices, frame_shape):
        coords = np.array([(landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0]) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _get_head_pose(self, landmarks, frame_shape):
        img_h, img_w, _ = frame_shape
        face_3d = np.array([[0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)
        face_2d = np.array([(landmarks[1].x * img_w, landmarks[1].y * img_h), (landmarks[152].x * img_w, landmarks[152].y * img_h), (landmarks[263].x * img_w, landmarks[263].y * img_h), (landmarks[33].x * img_w, landmarks[33].y * img_h), (landmarks[288].x * img_w, landmarks[288].y * img_h), (landmarks[58].x * img_w, landmarks[58].y * img_h)], dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
        if not success: return None
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles

class RealTimePredictor:
    def __init__(self, model_path='fatigue_lstm.pth'):
        self.sequence_length = 60
        self.feature_extractor = FeatureExtractor()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.model = LSTMModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.class_labels = ['ALERT', 'DROWSY']

    def run(self):
        print("Starting real-time ML fatigue detector. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            prediction_text = "STATUS: ANALYZING..."

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract features
                ear = self.feature_extractor._calculate_aspect_ratio(landmarks, self.feature_extractor.landmark_indices['right_eye'], frame.shape)
                mar = self.feature_extractor._calculate_aspect_ratio(landmarks, self.feature_extractor.landmark_indices['mouth'], frame.shape)
                head_angles = self.feature_extractor._get_head_pose(landmarks, frame.shape)

                if head_angles is not None:
                    features = [ear, mar, head_angles[0], head_angles[1], head_angles[2]]
                    self.feature_buffer.append(features)

                    # Check if the buffer is full
                    if len(self.feature_buffer) == self.sequence_length:
                        # Prepare sequence for the model
                        sequence_tensor = torch.tensor([list(self.feature_buffer)], dtype=torch.float32)
                        
                        # Make a prediction
                        with torch.no_grad():
                            output = self.model(sequence_tensor)
                            _, predicted_idx = torch.max(output, 1)
                            prediction = self.class_labels[predicted_idx.item()]
                        
                        prediction_text = f"STATUS: {prediction}"
                        
                        if prediction == 'DROWSY':
                            # Draw Drowsiness Alert
                            cv2.putText(frame, "DROWSINESS ALERT", (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)

            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Real-Time ML Fatigue Detector', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == '__main__':
    predictor = RealTimePredictor()
    predictor.run()
