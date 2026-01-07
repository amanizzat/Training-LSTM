import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
import sys
import os
import json

# Suppress TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
MODEL_PATH = Path('action_model.tflite')
LABELS_PATH = Path('action_labels.npy')
METADATA_PATH = Path('model_metadata.json')
FLUTTER_CONFIG_PATH = Path('flutter_config.json')

# Model parameters (will be auto-detected)
SEQUENCE_LENGTH = None
NUM_FEATURES = None
ACTIONS = None

# Confidence threshold - only show predictions above this
CONFIDENCE_THRESHOLD = 0.85
# ---------------------

def load_model_config():
    """Load model configuration from metadata files."""
    global SEQUENCE_LENGTH, NUM_FEATURES, ACTIONS
    
    print("\n" + "="*70)
    print("LOADING MODEL CONFIGURATION")
    print("="*70)
    
    # Try loading .npy first, then .txt
    npy_path = Path('action_labels.npy')
    txt_path = Path('action_labels.txt')
    
    if npy_path.exists():
        ACTIONS = np.load(str(npy_path))
        print(f"✓ Loaded {len(ACTIONS)} actions from {npy_path}: {list(ACTIONS)}")
    elif txt_path.exists():
        with open(txt_path, 'r') as f:
            ACTIONS = np.array([line.strip() for line in f if line.strip()])
        print(f"✓ Loaded {len(ACTIONS)} actions from {txt_path}: {list(ACTIONS)}")
    else:
        print(f"✗ ERROR: No labels file found (action_labels.npy or action_labels.txt)")
        sys.exit(1)
    
    # Try loading from model_metadata.json first
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            SEQUENCE_LENGTH = metadata.get('sequence_length', 30)
            NUM_FEATURES = metadata.get('num_features')
            extraction_mode = metadata.get('extraction_mode', 'Unknown')
            
            print(f"✓ Loaded configuration from: {METADATA_PATH}")
            print(f"  - Extraction mode: {extraction_mode}")
            print(f"  - Sequence length: {SEQUENCE_LENGTH}")
            print(f"  - Num features: {NUM_FEATURES}")
            return True
        except Exception as e:
            print(f"⚠ Warning: Could not load {METADATA_PATH}: {e}")
    
    # Try flutter_config.json as fallback
    if FLUTTER_CONFIG_PATH.exists():
        try:
            with open(FLUTTER_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            SEQUENCE_LENGTH = config.get('sequence_length', 30)
            NUM_FEATURES = config.get('num_features')
            
            print(f"✓ Loaded configuration from: {FLUTTER_CONFIG_PATH}")
            print(f"  - Sequence length: {SEQUENCE_LENGTH}")
            print(f"  - Num features: {NUM_FEATURES}")
            return True
        except Exception as e:
            print(f"⚠ Warning: Could not load {FLUTTER_CONFIG_PATH}: {e}")
    
    # Last resort: detect from model
    print("⚠ No metadata files found, will detect from TFLite model...")
    return False

def detect_config_from_model(interpreter):
    """Detect configuration from TFLite model input shape."""
    global SEQUENCE_LENGTH, NUM_FEATURES
    
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    # Shape is [batch, sequence_length, num_features]
    SEQUENCE_LENGTH = int(input_shape[1])
    NUM_FEATURES = int(input_shape[2])
    
    print(f"✓ Detected from model:")
    print(f"  - Sequence length: {SEQUENCE_LENGTH}")
    print(f"  - Num features: {NUM_FEATURES}")

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Processes an image with MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results, show_face=True):
    """Draws landmarks based on what features are being used."""
    # Draw face (only if face features are in model)
    if show_face and results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    
    # Draw pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    
    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    
    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints_standard(results):
    """
    Standard extraction: Pose + Hands = 258 features
    """
    pose = np.array([
        [res.x, res.y, res.z, res.visibility] 
        for res in results.pose_landmarks.landmark
    ]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    lh = np.array([
        [res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark
    ]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([
        [res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark
    ]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def extract_keypoints_maximal(results):
    """
    Maximal extraction: Pose + Face + Hands = 1662 features
    """
    pose = np.array([
        [res.x, res.y, res.z, res.visibility] 
        for res in results.pose_landmarks.landmark
    ]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([
        [res.x, res.y, res.z] 
        for res in results.face_landmarks.landmark
    ]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([
        [res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark
    ]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([
        [res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark
    ]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def extract_keypoints_adaptive(results, num_features):
    """
    Adaptive extraction: Auto-pad/truncate to match model
    """
    if num_features == 258:
        return extract_keypoints_standard(results)
    elif num_features == 1662:
        return extract_keypoints_maximal(results)
    else:
        keypoints = extract_keypoints_maximal(results)
        if len(keypoints) < num_features:
            keypoints = np.pad(keypoints, (0, num_features - len(keypoints)), 
                             mode='constant', constant_values=0)
        elif len(keypoints) > num_features:
            keypoints = keypoints[:num_features]
        return keypoints

def extract_keypoints(results, num_features):
    """Main extraction function that adapts to model requirements."""
    return extract_keypoints_adaptive(results, num_features)

# --- Main Function ---
def main():
    global SEQUENCE_LENGTH, NUM_FEATURES
    
    # Load configuration
    config_loaded = load_model_config()
    
    # Load TFLite model
    print(f"\nLoading TFLite model from: {MODEL_PATH}")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Detect config from model if not loaded from files
    if SEQUENCE_LENGTH is None or NUM_FEATURES is None:
        detect_config_from_model(interpreter)
    
    # Verify model shape matches config
    model_input_shape = input_details[0]['shape']
    expected_shape = [1, SEQUENCE_LENGTH, NUM_FEATURES]
    
    print(f"\n{'='*70}")
    print("MODEL CONFIGURATION")
    print(f"{'='*70}")
    print(f"Expected input shape: {expected_shape}")
    print(f"Actual model shape:   {model_input_shape.tolist()}")
    print(f"Output shape:         {output_details[0]['shape'].tolist()}")
    print(f"Actions ({len(ACTIONS)}): {list(ACTIONS)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    
    if (model_input_shape[1] != SEQUENCE_LENGTH or model_input_shape[2] != NUM_FEATURES):
        print(f"\n✗ ERROR: Model shape mismatch!")
        sys.exit(1)
    
    # Determine extraction mode for display
    if NUM_FEATURES == 258:
        extraction_mode = "Standard (Pose + Hands)"
        show_face = False
    elif NUM_FEATURES == 1662:
        extraction_mode = "Maximal (Pose + Face + Hands)"
        show_face = True
    else:
        extraction_mode = f"Custom ({NUM_FEATURES} features)"
        show_face = True
    
    print(f"Extraction mode:      {extraction_mode}")
    print(f"{'='*70}")
    
    # --- Real-time Detection Setup ---
    sequence_queue = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = ""
    confidence = 0.0
    
    print("\nStarting webcam feed...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to clear sequence buffer")
    print(f"\nNote: Only predictions with confidence >= {CONFIDENCE_THRESHOLD:.0%} will be shown")
    
    cap = cv2.VideoCapture(1)  # Change to 0, 1, 2, etc. for other cameras
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ ERROR: Could not open webcam.")
            sys.exit(1)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=(show_face)
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame from webcam.")
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw styled landmarks
            draw_styled_landmarks(image, results, show_face=show_face)
            
            # Extract keypoints
            keypoints = extract_keypoints(results, NUM_FEATURES)
            sequence_queue.append(keypoints)
            
            # Run prediction (only when queue is full)
            if len(sequence_queue) == SEQUENCE_LENGTH:
                try:
                    # Prepare input data
                    input_data = np.expand_dims(list(sequence_queue), axis=0)
                    input_data = input_data.astype(np.float32)
                    
                    # Set input tensor
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    
                    # Run inference
                    interpreter.invoke()
                    
                    # Get output tensor
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Get prediction
                    prediction_probabilities = output_data[0]
                    predicted_index = np.argmax(prediction_probabilities)
                    confidence = prediction_probabilities[predicted_index]
                    
                    # ONLY show prediction if confidence is HIGH
                    if confidence >= CONFIDENCE_THRESHOLD:
                        current_prediction = ACTIONS[predicted_index]
                    else:
                        current_prediction = "..."  # Low confidence, don't show
                
                except Exception as e:
                    print(f"Error during inference: {e}")
                    current_prediction = "Error"
                
                # Display prediction on screen
                if confidence >= CONFIDENCE_THRESHOLD:
                    # High confidence - show green bar
                    cv2.rectangle(image, (0,0), (640, 60), (0, 128, 0), -1)
                    cv2.putText(image, f"PREDICTION: {current_prediction}", 
                               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Confidence: {confidence:.1%}", 
                               (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Low confidence - show gray bar with "..." 
                    cv2.rectangle(image, (0,0), (640, 60), (100, 100, 100), -1)
                    cv2.putText(image, f"Detecting... (confidence: {confidence:.1%})", 
                               (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            
            else:
                # Display waiting message
                cv2.rectangle(image, (0,0), (640, 40), (100, 100, 100), -1)
                cv2.putText(image, f"Collecting frames... ({len(sequence_queue)}/{SEQUENCE_LENGTH})", 
                           (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display extraction mode info
            cv2.putText(image, f"Mode: {extraction_mode}", 
                       (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('Sign Language Recognition', image)
            
            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sequence_queue.clear()
                print("Sequence buffer cleared.")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nWebcam feed stopped.")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)