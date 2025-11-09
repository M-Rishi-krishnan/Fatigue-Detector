# preprocess_utarldd.py
import cv2
import mediapipe as mp
import numpy as np
import os
import glob

class FeatureExtractor:
    def __init__(self, config_path="config.json"):
        pass

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

def process_videos(dataset_path, output_path, sequence_length=60):
    feature_extractor = FeatureExtractor()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    labels = {"alert": "Vigilant", "drowsy": "DrowsyState"}

    for label_class, folder_name in labels.items():
        class_path = os.path.join(dataset_path, folder_name)
        dest_path = os.path.join(output_path, label_class)
        os.makedirs(dest_path, exist_ok=True)
        
        video_files = glob.glob(os.path.join(class_path, '**', '*.avi'), recursive=True)
        print(f"Found {len(video_files)} videos for class '{label_class}'")

        for video_file in video_files:
            print(f"Processing {video_file}...")
            cap = cv2.VideoCapture(video_file)
            features_sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    ear = feature_extractor._calculate_aspect_ratio(landmarks, [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], frame.shape) # Using standard right eye indices
                    mar = feature_extractor._calculate_aspect_ratio(landmarks, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0, 270, 269, 267, 409, 291], frame.shape) # Using standard mouth indices
                    head_angles = feature_extractor._get_head_pose(landmarks, frame.shape)

                    if head_angles is not None:
                        features = [ear, mar, head_angles[0], head_angles[1], head_angles[2]]
                        features_sequence.append(features)

            cap.release()

            # Segment the full video's features into sequences of `sequence_length`
            num_sequences = len(features_sequence) // sequence_length
            for i in range(num_sequences):
                start = i * sequence_length
                end = start + sequence_length
                sequence = features_sequence[start:end]
                
                # Save the sequence
                sequence_name = f"{os.path.basename(video_file).split('.')[0]}_{i}.npy"
                np.save(os.path.join(dest_path, sequence_name), np.array(sequence))

    face_mesh.close()
    print("Finished processing all videos.")

if __name__ == '__main__':
    dataset_root_path = 'C:/path/to/your/UTA-RLDD'
    output_data_path = 'fatigue_data'
    
    process_videos(dataset_root_path, output_data_path, sequence_length=60)

