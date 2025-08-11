import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_mar(landmarks, frame_shape):
    mouth_indices = [61, 291, 0, 17, 37, 267, 39, 269] 
    coords = np.array([(landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0]) for i in mouth_indices])
    
    v1 = np.linalg.norm(coords[2] - coords[6])
    v2 = np.linalg.norm(coords[3] - coords[5])
    h = np.linalg.norm(coords[0] - coords[4])
    
    return (v1 + v2) / (2.0 * h + 1e-6)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

print("--- Mouth Aspect Ratio Calibration ---")
print("This script will help you find the perfect MAR threshold for yawn detection.")
time.sleep(2)

print("\nPHASE 1: Please look at the camera with your mouth comfortably CLOSED for 5 seconds.")
closed_mar_values = []
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret: continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        mar = calculate_mar(landmarks, frame.shape)
        closed_mar_values.append(mar)
    
    cv2.putText(frame, "STATUS: Keep Mouth CLOSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(5)

avg_closed_mar = np.mean(closed_mar_values) if closed_mar_values else 0
print(f"-> Average MAR with mouth closed: {avg_closed_mar:.4f}")

print("\nPHASE 2: Now, please perform a big, wide YAWN and hold it for 5 seconds.")
open_mar_values = []
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret: continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        mar = calculate_mar(landmarks, frame.shape)
        open_mar_values.append(mar)

    cv2.putText(frame, "STATUS: Perform a big YAWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(5)

max_open_mar = np.max(open_mar_values) if open_mar_values else 0
print(f"-> Maximum MAR during yawn: {max_open_mar:.4f}")

cap.release()
cv2.destroyAllWindows()

if avg_closed_mar > 0 and max_open_mar > avg_closed_mar:
    recommended_threshold = avg_closed_mar + (max_open_mar - avg_closed_mar) / 2
    print("\n--- CALIBRATION COMPLETE ---")
    print(f"\nYour recommended 'mar_threshold' is: {recommended_threshold:.4f}")
    print("Please update this value in your 'config.json' file.")
else:
    print("\n--- CALIBRATION FAILED ---")
    print("Could not get reliable readings. Please try running the script again.")

