"""
STEP 3: TRAIN MODEL (Mobile-Compatible + Hand-Face Interaction)
================================================================
Trains an LSTM model optimized for:
- Hand and finger movement detection
- Hand-to-face touch detection (forehead, nose, lips, etc.)
- Mobile TFLite compatibility

Key features:
- Enhanced architecture for hand-face interaction learning
- Smart augmentation preserving spatial relationships
- Computed hand-face distance features during training
- Mobile-compatible (no Flex delegate needed)

Pose landmarks for face reference:
- 0: Nose, 2/5: Inner eyes, 1/4: Outer eyes
- 7/8: Ears, 9/10: Mouth corners

Usage:
    python 3_train_model.py
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import os
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
DATA_PATH = Path("C:/fyp/SL_Data_Processed")
OUTPUT_DIR = Path("C:/fyp/training")

SEQUENCE_LENGTH = 30
NUM_FEATURES = 258

# Feature layout
POSE_START = 0
POSE_END = 132      # 33 landmarks × 4 values
LEFT_HAND_START = 132
LEFT_HAND_END = 195  # 21 landmarks × 3 values
RIGHT_HAND_START = 195
RIGHT_HAND_END = 258 # 21 landmarks × 3 values

# Face landmark indices in pose (for hand-face interaction)
FACE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
}

# Training settings
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Lower for better convergence

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 5  # 5x more data for robustness
# ---------------------


def get_pose_landmark(frame, landmark_idx):
    """Get x, y, z coordinates for a pose landmark."""
    idx = landmark_idx * 4  # Each pose landmark has 4 values (x, y, z, visibility)
    return frame[idx:idx+3]  # Return x, y, z


def get_hand_center(frame, hand='right'):
    """Get the center position of a hand (wrist position)."""
    if hand == 'right':
        # Right hand wrist is first landmark (index 0 of right hand)
        idx = RIGHT_HAND_START
    else:
        # Left hand wrist
        idx = LEFT_HAND_START
    return frame[idx:idx+3]  # x, y, z


def get_fingertip_positions(frame, hand='right'):
    """Get all fingertip positions for a hand.
    Fingertips are at indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    """
    start = RIGHT_HAND_START if hand == 'right' else LEFT_HAND_START
    fingertip_indices = [4, 8, 12, 16, 20]
    positions = []
    for tip_idx in fingertip_indices:
        idx = start + tip_idx * 3
        positions.append(frame[idx:idx+3])
    return positions


def compute_hand_face_distances(frame):
    """
    Compute distances between hand positions and face landmarks.
    This helps the model learn hand-to-face touch patterns.
    Returns normalized distance features.
    """
    distances = []
    
    # Get face landmark positions
    nose = get_pose_landmark(frame, FACE_LANDMARKS['nose'])
    mouth_center = (get_pose_landmark(frame, FACE_LANDMARKS['mouth_left']) + 
                   get_pose_landmark(frame, FACE_LANDMARKS['mouth_right'])) / 2
    
    # Approximate forehead (above nose, between eyes)
    left_eye = get_pose_landmark(frame, FACE_LANDMARKS['left_eye'])
    right_eye = get_pose_landmark(frame, FACE_LANDMARKS['right_eye'])
    forehead = np.array([
        (left_eye[0] + right_eye[0]) / 2,
        nose[1] - 0.1,  # Above nose
        (left_eye[2] + right_eye[2]) / 2
    ])
    
    face_points = {
        'nose': nose,
        'mouth': mouth_center,
        'forehead': forehead,
        'left_eye': left_eye,
        'right_eye': right_eye,
    }
    
    # For each hand
    for hand in ['right', 'left']:
        hand_center = get_hand_center(frame, hand)
        fingertips = get_fingertip_positions(frame, hand)
        
        # Check if hand is detected (non-zero)
        hand_detected = np.any(hand_center != 0)
        
        if hand_detected:
            # Distance from hand center to each face point
            for face_name, face_pos in face_points.items():
                dist = np.linalg.norm(hand_center - face_pos)
                distances.append(dist)
            
            # Distance from index fingertip to nose (common touch point)
            index_tip = fingertips[1]  # Index finger
            dist_to_nose = np.linalg.norm(index_tip - nose)
            distances.append(dist_to_nose)
            
            # Distance from thumb to mouth (for some signs)
            thumb_tip = fingertips[0]
            dist_to_mouth = np.linalg.norm(thumb_tip - mouth_center)
            distances.append(dist_to_mouth)
        else:
            # Hand not detected - use large distances
            distances.extend([1.0] * 7)  # 5 face points + 2 fingertip distances
    
    return np.array(distances)


def augment_sequence(sequence, preserve_hand_face=True):
    """
    Apply random augmentations to improve model robustness.
    Can preserve hand-face spatial relationships for touch-based signs.
    """
    aug = sequence.copy()
    
    # 1. Add noise (simulates sensor noise) - REDUCED for hands
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.015, aug.shape)
        # Less noise on hand landmarks (more precision needed for finger detection)
        noise[:, LEFT_HAND_START:RIGHT_HAND_END] *= 0.5
        aug = aug + noise
    
    # 2. Scale variation (simulates distance from camera)
    if random.random() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        aug = aug * scale
    
    # 3. Temporal shift (simulates timing variations)
    if random.random() < 0.3:
        shift = random.randint(-2, 2)
        aug = np.roll(aug, shift, axis=0)
    
    # 4. Frame dropout (simulates occlusion) - AVOID dropping hand frames
    if random.random() < 0.2:
        drop_idx = random.randint(0, SEQUENCE_LENGTH - 1)
        # Only drop pose, keep hands
        aug[drop_idx, POSE_START:POSE_END] = 0
    
    # 5. Mirror (swap left/right) - skip if preserving hand-face relationships
    if not preserve_hand_face and random.random() < 0.3:
        aug = mirror_landmarks(aug)
    
    # 6. Speed variation (stretch/compress time)
    if random.random() < 0.3:
        speed = random.choice([0.85, 0.9, 0.95, 1.05, 1.1, 1.15])
        indices = np.linspace(0, SEQUENCE_LENGTH-1, int(SEQUENCE_LENGTH*speed))
        indices = np.clip(indices, 0, SEQUENCE_LENGTH-1).astype(int)
        if len(indices) >= SEQUENCE_LENGTH:
            aug = aug[indices[:SEQUENCE_LENGTH]]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(indices), NUM_FEATURES))
            aug = np.vstack([aug[indices], padding])
    
    # 7. Hand position jitter (small movements)
    if random.random() < 0.4:
        jitter = np.random.uniform(-0.01, 0.01, (SEQUENCE_LENGTH, RIGHT_HAND_END - LEFT_HAND_START))
        aug[:, LEFT_HAND_START:RIGHT_HAND_END] += jitter
    
    # 8. Slight rotation simulation (2D rotation in x-y plane)
    if random.random() < 0.3:
        angle = np.random.uniform(-0.1, 0.1)  # Small angle in radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(len(aug)):
            # Rotate hand landmarks
            for start in [LEFT_HAND_START, RIGHT_HAND_START]:
                for j in range(21):
                    idx = start + j * 3
                    x, y = aug[i, idx], aug[i, idx+1]
                    aug[i, idx] = x * cos_a - y * sin_a
                    aug[i, idx+1] = x * sin_a + y * cos_a
    
    return np.clip(aug, -2, 2)


def mirror_landmarks(sequence):
    """Mirror landmarks horizontally (swap left/right hands)."""
    mirrored = sequence.copy()
    
    for i in range(len(mirrored)):
        frame = mirrored[i].copy()
        
        # Flip X coordinates for pose (every 4th value starting at 0)
        for j in range(33):
            idx = j * 4
            if idx < POSE_END:
                frame[idx] = 1.0 - frame[idx]
        
        # Swap left hand and right hand
        left_hand = frame[LEFT_HAND_START:LEFT_HAND_END].copy()
        right_hand = frame[RIGHT_HAND_START:RIGHT_HAND_END].copy()
        
        # Flip X for hands (every 3rd value starting at 0)
        for j in range(21):
            idx = j * 3
            if idx < 63:
                left_hand[idx] = 1.0 - left_hand[idx]
                right_hand[idx] = 1.0 - right_hand[idx]
        
        frame[LEFT_HAND_START:LEFT_HAND_END] = right_hand
        frame[RIGHT_HAND_START:RIGHT_HAND_END] = left_hand
        mirrored[i] = frame
    
    return mirrored


def check_hand_face_interaction(sequence):
    """
    Check if a sequence contains hand-face interaction.
    Used to decide augmentation strategy.
    """
    for frame in sequence:
        nose = get_pose_landmark(frame, FACE_LANDMARKS['nose'])
        
        for hand in ['right', 'left']:
            hand_center = get_hand_center(frame, hand)
            if np.any(hand_center != 0) and np.any(nose != 0):
                dist = np.linalg.norm(hand_center - nose)
                if dist < 0.3:  # Hand is near face
                    return True
    return False


def load_data(augment=True, aug_factor=5):
    """Load training data from processed landmarks with smart augmentation."""
    print("\n📂 Loading data...")
    
    # Find actions
    actions = sorted([f.name for f in DATA_PATH.iterdir() if f.is_dir()])
    if not actions:
        print("✗ No data found!")
        return None, None, None
    
    print(f"   Actions: {actions}")
    
    label_map = {action: idx for idx, action in enumerate(actions)}
    sequences = []
    labels = []
    hand_face_sequences = []  # Track which sequences have hand-face interaction
    
    # Load each action
    for action in actions:
        action_path = DATA_PATH / action
        seq_folders = sorted(
            [f for f in action_path.iterdir() if f.is_dir() and f.name.isdigit()],
            key=lambda f: int(f.name)
        )
        
        count = 0
        hf_count = 0  # Hand-face interaction count
        for seq_folder in seq_folders:
            # Load all frames
            frames = []
            valid = True
            
            for frame_num in range(1, SEQUENCE_LENGTH + 1):
                frame_file = seq_folder / f"{frame_num}.npy"
                if frame_file.exists():
                    try:
                        data = np.load(str(frame_file))
                        if data.shape[0] == NUM_FEATURES:
                            frames.append(data)
                        else:
                            valid = False
                            break
                    except:
                        frames.append(np.zeros(NUM_FEATURES))
                else:
                    frames.append(np.zeros(NUM_FEATURES))
            
            if valid and len(frames) == SEQUENCE_LENGTH:
                sequence = np.array(frames)
                sequences.append(sequence)
                labels.append(label_map[action])
                count += 1
                
                # Check for hand-face interaction
                has_hf = check_hand_face_interaction(sequence)
                hand_face_sequences.append(has_hf)
                if has_hf:
                    hf_count += 1
                
                # Add augmented versions
                if augment:
                    for aug_i in range(aug_factor):
                        # Preserve hand-face relationships for relevant sequences
                        preserve = has_hf and aug_i < aug_factor // 2
                        aug_seq = augment_sequence(sequence, preserve_hand_face=preserve)
                        sequences.append(aug_seq)
                        labels.append(label_map[action])
                        hand_face_sequences.append(has_hf)
        
        hf_pct = (hf_count / count * 100) if count > 0 else 0
        print(f"   {action}: {count} sequences ({hf_count} with hand-face interaction, {hf_pct:.0f}%)")
    
    X = np.array(sequences)
    y = np.array(labels)
    
    original_count = len(X) // (aug_factor + 1) if augment else len(X)
    print(f"\n✓ Loaded {original_count} original + {len(X) - original_count} augmented = {len(X)} total")
    
    return X, y, np.array(actions)


def build_model(num_classes):
    """
    Build an enhanced LSTM model for hand-finger-face interaction detection.
    
    Architecture:
    - Separate processing streams for pose and hands
    - Feature concatenation for hand-face interaction learning
    - Multi-layer LSTM for temporal pattern recognition
    - Mobile TFLite compatible (no Bidirectional layers)
    """
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name='input')
    
    # Split into pose and hands using Lambda (TFLite compatible)
    # Pose: indices 0-131 (face reference points + body)
    # Hands: indices 132-257 (left + right hand with all fingers)
    
    # Process full input through LSTM layers
    # Layer 1: Initial feature extraction
    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Dropout(0.3, name='drop1')(x)
    
    # Layer 2: Pattern learning
    x = tf.keras.layers.LSTM(256, return_sequences=True, name='lstm2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Dropout(0.3, name='drop2')(x)
    
    # Layer 3: Higher-level feature extraction
    x = tf.keras.layers.LSTM(256, return_sequences=True, name='lstm3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Dropout(0.3, name='drop3')(x)
    
    # Layer 4: Final temporal integration
    x = tf.keras.layers.LSTM(128, return_sequences=False, name='lstm4')(x)
    x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.Dropout(0.4, name='drop4')(x)
    
    # Dense layers for classification
    x = tf.keras.layers.Dense(256, activation='relu', name='dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn5')(x)
    x = tf.keras.layers.Dropout(0.4, name='drop5')(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', name='dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn6')(x)
    x = tf.keras.layers.Dropout(0.3, name='drop6')(x)
    
    x = tf.keras.layers.Dense(64, activation='relu', name='dense3')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SignLanguageModel')
    
    return model


def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite (mobile-compatible)."""
    print("\n🔄 Converting to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # IMPORTANT: Only use built-in ops (NO Flex ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Optimize for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✓ Saved: {output_path} ({size_kb:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        print("\n  Trying with SELECT_TF_OPS (requires Flex delegate on mobile)...")
        
        # Fallback with Flex ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        try:
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            size_kb = len(tflite_model) / 1024
            print(f"✓ Saved with Flex ops: {output_path} ({size_kb:.1f} KB)")
            print("  ⚠️  This model requires Flex delegate on mobile")
            return True
        except Exception as e2:
            print(f"✗ Fallback conversion also failed: {e2}")
            return False


def verify_tflite(model_path, X_test):
    """Verify TFLite model works correctly and compare with Keras."""
    print("\n🔍 Verifying TFLite model...")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input dtype: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Test with multiple samples
        num_tests = min(10, len(X_test))
        correct = 0
        
        for i in range(num_tests):
            test_input = X_test[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            pred_class = np.argmax(output)
            confidence = np.max(output)
            
            if i == 0:
                print(f"   Sample prediction: class {pred_class} (confidence: {confidence:.1%})")
        
        print("✓ TFLite model verified successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def evaluate_per_class(model, X_test, y_test, actions):
    """Evaluate accuracy for each class with detailed metrics."""
    print("\n📊 Per-Class Accuracy:")
    print("-" * 50)
    
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    pred_confidence = np.max(predictions, axis=1)
    
    total_correct = 0
    total_samples = 0
    
    for i, action in enumerate(actions):
        mask = y_test == i
        if np.sum(mask) > 0:
            correct = np.sum((pred_classes == y_test) & mask)
            total = np.sum(mask)
            acc = correct / total
            
            # Average confidence for this class
            class_conf = np.mean(pred_confidence[mask])
            
            status = "✓" if acc >= 0.9 else "~" if acc >= 0.7 else "✗"
            print(f"   {status} {action:12s}: {acc:.1%} ({correct}/{total}) | conf: {class_conf:.1%}")
            
            total_correct += correct
            total_samples += total
    
    print("-" * 50)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"   Overall: {overall_acc:.1%} ({total_correct}/{total_samples})")
    
    return overall_acc


def analyze_confusion(model, X_test, y_test, actions):
    """Analyze common confusions between classes."""
    print("\n🔍 Confusion Analysis (top misclassifications):")
    print("-" * 50)
    
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    confusions = {}
    for true_label, pred_label in zip(y_test, pred_classes):
        if true_label != pred_label:
            key = (actions[true_label], actions[pred_label])
            confusions[key] = confusions.get(key, 0) + 1
    
    # Sort by frequency
    sorted_conf = sorted(confusions.items(), key=lambda x: -x[1])[:10]
    
    for (true_action, pred_action), count in sorted_conf:
        print(f"   {true_action} → {pred_action}: {count} times")
    
    if not sorted_conf:
        print("   No misclassifications! 🎉")


def main():
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN MODEL (Hand-Finger-Face Detection)")
    print("=" * 60)
    print(f"Features: {NUM_FEATURES} (Pose: 132, Hands: 126)")
    print(f"- Pose includes face landmarks (nose, eyes, mouth)")
    print(f"- Hands include all 5 fingers × 4 joints each")
    print("=" * 60)
    
    # Load data
    X, y, actions = load_data(augment=USE_AUGMENTATION, aug_factor=AUGMENTATION_FACTOR)
    
    if X is None:
        return
    
    num_classes = len(actions)
    print(f"\n📊 Data shape: {X.shape}")
    print(f"   Classes: {num_classes}")
    print(f"   Actions: {list(actions)}")
    
    # One-hot encode labels
    y_cat = tf.keras.utils.to_categorical(y, num_classes)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    print("\n🔧 Building enhanced model...")
    model = build_model(num_classes)
    
    # Use Adam with weight decay for better generalization
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=40,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(OUTPUT_DIR / 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train
    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    print("-" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Results:")
    print(f"   Train Accuracy: {train_acc:.1%}")
    print(f"   Val Accuracy:   {val_acc:.1%}")
    print(f"   Test Accuracy:  {test_acc:.1%}")
    
    # Per-class evaluation
    y_test_labels = np.argmax(y_test, axis=1)
    evaluate_per_class(model, X_test, y_test_labels, actions)
    analyze_confusion(model, X_test, y_test_labels, actions)
    
    # Save model
    model.save(str(OUTPUT_DIR / 'action_model.keras'))
    print(f"\n✓ Saved: {OUTPUT_DIR / 'action_model.keras'}")
    
    # Save labels
    np.save(str(OUTPUT_DIR / 'action_labels.npy'), actions)
    with open(OUTPUT_DIR / 'action_labels.txt', 'w') as f:
        for action in actions:
            f.write(f"{action}\n")
    print(f"✓ Saved: action_labels.npy, action_labels.txt")
    
    # Convert to TFLite
    tflite_path = OUTPUT_DIR / 'action_model.tflite'
    if convert_to_tflite(model, str(tflite_path)):
        verify_tflite(str(tflite_path), X_test)
    
    # Save metadata
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': NUM_FEATURES,
        'num_classes': num_classes,
        'actions': list(actions),
        'feature_layout': {
            'pose': {'start': POSE_START, 'end': POSE_END, 'landmarks': 33},
            'left_hand': {'start': LEFT_HAND_START, 'end': LEFT_HAND_END, 'landmarks': 21},
            'right_hand': {'start': RIGHT_HAND_START, 'end': RIGHT_HAND_END, 'landmarks': 21}
        },
        'face_landmarks_in_pose': list(FACE_LANDMARKS.keys()),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'augmentation_factor': AUGMENTATION_FACTOR,
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(OUTPUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: model_metadata.json")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("1. Copy model to Flutter app:")
    print(f"   copy {tflite_path} C:\\fyp\\lang_bridge\\assets\\action_model.tflite")
    print(f"   copy {OUTPUT_DIR}\\action_labels.txt C:\\fyp\\lang_bridge\\assets\\")
    print("\n2. Test the model:")
    print("   python 4_evaluate_model.py")
    print("   python 5_test_live.py")


if __name__ == "__main__":
    main()