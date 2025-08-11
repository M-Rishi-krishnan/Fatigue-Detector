# Real-Time Fatigue Detector

This project is a real-time fatigue monitoring system developed in Python. It utilizes a standard webcam to analyze a user's facial landmarks and identify key signs of fatigue, including drowsiness, excessive yawning, and high blink rates, providing immediate alerts.

## Features

- **Multi-Factor Fatigue Detection:** Monitors eye closure (drowsiness), mouth opening (yawning), and blink frequency for a comprehensive analysis.
- **Real-Time Alerts:** Provides both visual and auditory feedback to the user when fatigue is detected.
- **User Calibration:** Includes a dedicated utility to calibrate yawn detection for individual users, improving accuracy.
- **Configurable:** All sensitivity thresholds are managed in an external `config.json` file for easy tuning.

## Technologies Used

- **Python**
- **OpenCV:** For real-time video capture and image processing.
- **MediaPipe:** For high-fidelity facial landmark detection.
- **Pygame:** For handling audio alerts.
- **NumPy:** For numerical operations.

## Setup and Usage

### Prerequisites

- Python 3.8+
- A webcam

### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/M-Rishi-Krishnan/Fatigue-Detector.git
    cd Fatigue-Detector
    ```

2.  **Install dependencies:**
    ```
    pip install opencv-python mediapipe numpy pygame
    ```

3.  **Add an alert sound:**
    Place a short `.wav` file named `alert.wav` in the project root directory.

### Running the Application

1.  **Calibrate for Yawn Detection (Important):**
    Run the calibration script and follow the on-screen instructions. This will give you a personalized `mar_threshold`.
    ```
    python calibrate.py
    ```
    Update the `mar_threshold` value in `config.json` with the recommended value from the script.

2.  **Run the Main Detector:**
    ```
    python main.py
    ```
    The application will start. Press 'q' to quit.
