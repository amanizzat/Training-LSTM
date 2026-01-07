import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort # Import ONNX Runtime
from collections import deque
from pathlib import Path
import sys

# --- Configuration ---
MODEL_PATH = Path('action_model_improved.onnx') # Use the ONNX file
LABELS_PATH = Path('action_labels.npy')
SEQUENCE_LENGTH = 30
NUM_FEATURES = 258 
CONFIDENCE_THRESHOLD = 0.6
# ---------------------

# Load Actions
try:
    ACTIONS = np.load(str(LABELS_PATH))
except:
    print("Error loading labels.")
    sys.exit()

# --- ONNX Setup ---
print(f"Loading ONNX model from {MODEL_PATH}...")
try:
    # Create Inference Session
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✓ Model loaded successfully.")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit()

# --- MediaPipe Setup (Same as before) ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    # Pose (132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Left Hand (63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Right Hand (63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- Real-time Detection ---
sequence = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "..."
confidence = 0.0

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks (Simple visualization)
        if results.left_hand_landmarks:
             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            # Prepare input for ONNX (Batch, Seq, Feat)
            # ONNX expects float32 specifically
            input_data = np.expand_dims(list(sequence), axis=0).astype(np.float32)

            # Run Inference
            res = session.run([output_name], {input_name: input_data})[0]
            
            # Process Result
            predicted_index = np.argmax(res[0])
            confidence = res[0][predicted_index]

            if confidence > CONFIDENCE_THRESHOLD:
                current_prediction = ACTIONS[predicted_index]
            else:
                current_prediction = "..."

        # Display
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"PRED: {current_prediction} ({confidence:.2f})", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('ONNX Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()