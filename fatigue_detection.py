import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import pygame

class FatigueDetector:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        pygame.mixer.init()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.alert_sound = None
        if os.path.exists(self.config['sound_path']):
            self.alert_sound = pygame.mixer.Sound(self.config['sound_path'])

        self.counters = {
            'eye_closed_frames': 0,
            'yawn_frames': 0,
            'blink_counter': 0,
            'blinks_in_current_minute': 0,
            'yawn_counter': 0
        }
        self.yawn_alert_sound_played = False
        self.start_time = time.time()
        self.landmark_indices = self.config['landmark_indices']

    def _calculate_aspect_ratio(self, landmarks, indices, frame_shape, is_mar=False):
        coords = np.array([(landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0]) for i in indices])
        if is_mar:
            v1 = np.linalg.norm(coords[2] - coords[6])
            v2 = np.linalg.norm(coords[3] - coords[5])
            h = np.linalg.norm(coords[0] - coords[4])
        else:
            v1 = np.linalg.norm(coords[1] - coords[5])
            v2 = np.linalg.norm(coords[2] - coords[4])
            h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _draw_alert(self, frame, message, play_sound=True):
        """Draws the visual alert and optionally plays sound."""
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)
        
        if play_sound and self.alert_sound and not pygame.mixer.get_busy():
            self.alert_sound.play()

    def run(self):
        print("Starting webcam. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = self.face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_height, frame_width, _ = frame.shape

            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 60:
                self.start_time = time.time()
                self.counters['blinks_in_current_minute'] = 0
                self.counters['yawn_counter'] = 0
                self.yawn_alert_sound_played = False

            alert_message = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                avg_ear = (self._calculate_aspect_ratio(landmarks, self.landmark_indices['left_eye'], (frame_height, frame_width)) +
                           self._calculate_aspect_ratio(landmarks, self.landmark_indices['right_eye'], (frame_height, frame_width))) / 2.0
                
                mar = self._calculate_aspect_ratio(landmarks, self.landmark_indices['mouth'], (frame_height, frame_width), is_mar=True)

                if avg_ear < self.config['ear_threshold']:
                    self.counters['eye_closed_frames'] += 1
                else:
                    if self.counters['eye_closed_frames'] >= self.config['consecutive_frames']['blink']:
                        self.counters['blink_counter'] += 1
                        self.counters['blinks_in_current_minute'] += 1
                    self.counters['eye_closed_frames'] = 0
                
                if mar > self.config['mar_threshold']:
                    self.counters['yawn_frames'] += 1
                else:
                    if self.counters['yawn_frames'] >= self.config['consecutive_frames']['yawn']:
                        self.counters['yawn_counter'] += 1
                    self.counters['yawn_frames'] = 0

                is_alert = False
                # Priority 1: Drowsiness (Continuous Sound)
                if self.counters['eye_closed_frames'] > self.config['consecutive_frames']['drowsiness']:
                    self._draw_alert(frame, "DROWSINESS ALERT!", play_sound=True)
                    is_alert = True
                
                # Priority 2: Excessive Yawning (Sound plays once)
                elif self.counters['yawn_counter'] >= self.config['yawn_threshold']:
                    if not self.yawn_alert_sound_played:
                        self._draw_alert(frame, "EXCESSIVE YAWNING!", play_sound=True)
                        self.yawn_alert_sound_played = True
                    else:
                        self._draw_alert(frame, "EXCESSIVE YAWNING!", play_sound=False)
                    is_alert = True

                # Priority 3: High Blink Rate (Visual only)
                else:
                    current_bpm = (self.counters['blinks_in_current_minute'] / elapsed_time) * 60 if elapsed_time > 1 else 0
                    if current_bpm > self.config['high_blink_rate_threshold'] and elapsed_time > 10:
                        self._draw_alert(frame, "High Blink Rate!", play_sound=False)
                        is_alert = True

                # Stop any playing sound if no alert condition is met
                if not is_alert and pygame.mixer.get_busy():
                    pygame.mixer.stop()

                cv2.putText(frame, f"Blinks: {self.counters['blink_counter']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawns (in last min): {self.counters['yawn_counter']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Fatigue Detection System', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == "__main__":
    detector = FatigueDetector()
    detector.run()
