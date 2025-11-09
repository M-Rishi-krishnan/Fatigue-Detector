# data_collector.py
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
from scipy.spatial.transform import Rotation as R
from collections import deque

class FatigueDetectorWithPosePER:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.counters = {
            'eye_closed_frames': 0,
            'perclos_buffer': deque(maxlen=60),  # Buffer for the last 60 frames/seconds
            'blink_counter': 0
        }
        self.landmark_indices = self.config['landmark_indices']

    def _calculate_aspect_ratio(self, landmarks, indices, frame_shape):
        coords = np.array([(landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0]) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _get_head_pose(self, landmarks, frame_shape):
        img_h, img_w, _ = frame_shape
        face_3d = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0, -150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[33].x * img_w, landmarks[33].y * img_h),
            (landmarks[288].x * img_w, landmarks[288].y * img_h),
            (landmarks[58].x * img_w, landmarks[58].y * img_h),
        ], dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)

        if not success:
            return None
            
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles

    def calculate_perclos(self):
        if not self.counters['perclos_buffer']:
            return 0.0
        return sum(self.counters['perclos_buffer']) / len(self.counters['perclos_buffer'])

    def run(self):
        print("Starting webcam with Head Pose and PERCLOS. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            alert_message = ""

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                ear = self._calculate_aspect_ratio(landmarks, self.landmark_indices['right_eye'], frame.shape)
                head_angles = self._get_head_pose(landmarks, frame.shape)

                is_eye_closed = 1 if ear < self.config['ear_threshold'] else 0
                self.counters['perclos_buffer'].append(is_eye_closed)
                perclos = self.calculate_perclos()

                if perclos > self.config['perclos_threshold']:
                    alert_message = "DROWSINESS ALERT"
                
                if head_angles is not None and head_angles[0] > self.config['head_pitch_threshold']:
                    alert_message = "DROWSINESS ALERT"
                    
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if head_angles is not None:
                    cv2.putText(frame, f"Head Pitch: {head_angles[0]:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if alert_message:
                cv2.putText(frame, alert_message, (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)

            cv2.imshow('Enhanced Fatigue Detection', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

class DataCollector(FatigueDetectorWithPosePER):
    def __init__(self, config_path="config.json"):
        super().__init__(config_path)
        self.sequence_length = 60
        self.data_path = "fatigue_data"
        os.makedirs(os.path.join(self.data_path, 'alert'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, 'drowsy'), exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(0)
        collecting = False
        sequence = []
        label = ""
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'): break
            if not collecting and (key == ord('a') or key == ord('d')):
                collecting = True
                sequence = []
                label = 'alert' if key == ord('a') else 'drowsy'
                print(f"Starting collection for: {label}")

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear = self._calculate_aspect_ratio(landmarks, self.landmark_indices['right_eye'], frame.shape)
                mar = self._calculate_aspect_ratio(landmarks, self.landmark_indices['mouth'], frame.shape)
                head_angles = self._get_head_pose(landmarks, frame.shape)

                if collecting and head_angles is not None:
                    features = [ear, mar, head_angles[0], head_angles[1], head_angles[2]]
                    sequence.append(features)
                    
                    if len(sequence) == self.sequence_length:
                        filepath = os.path.join(self.data_path, label, f"{int(time.time())}.npy")
                        np.save(filepath, np.array(sequence))
                        print(f"Saved sequence to {filepath}")
                        collecting = False

            if collecting:
                cv2.putText(frame, f"RECORDING: {label.upper()} ({len(sequence)}/{self.sequence_length})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Data Collector', frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    collector.run()
