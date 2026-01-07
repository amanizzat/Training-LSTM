"""
MODEL OPTIMIZATION SCRIPT
=========================
Retrains the model focusing on problem signs identified from diagnostic testing.

Key optimizations:
1. Contrastive loss for confused sign pairs (C/D, 2/3, etc.)
2. Focal loss to focus on hard examples
3. Enhanced augmentation for signs with low hand detection
4. Better Idle detection (no-hand classifier)
5. Class weighting based on confusion patterns

Problem signs identified:
- 2 confused with 3
- 5 confused with 3  
- B confused with 4/E
- C confused with D (100% confusion!)
- Biru confused with D
- Hijau confused with Merah/Kuning
- Hitam confused with D
- Selamat petang confused with Selamat malam (100%)
- Terima kasih confused with Merah (100%)
- Sama-sama confused with Merah (100%)
- Idle confused with B/2/4

Root causes identified:
- Low hand detection in training data for many signs
- Similar visual features between confused pairs
- Greetings have similar motion patterns

Usage:
    python 6_optimize_model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import random
import os
import json
from datetime import datetime

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
POSE_END = 132
LEFT_HAND_START = 132
LEFT_HAND_END = 195
RIGHT_HAND_START = 195
RIGHT_HAND_END = 258

# Training settings
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0003

# Problem signs and their confusions (from diagnostic)
CONFUSION_PAIRS = [
    ("2", "3"),
    ("5", "3"),
    ("B", "4"),
    ("B", "E"),
    ("C", "D"),
    ("Biru", "D"),
    ("Hijau", "Merah"),
    ("Hijau", "Kuning"),
    ("Hitam", "D"),
    ("Selamat petang", "Selamat malam"),
    ("Terima kasih", "Merah"),
    ("Sama-sama", "Merah"),
]

# Signs with low accuracy that need more focus
PROBLEM_SIGNS = [
    "2", "5", "B", "C", "E", "Biru", "Hijau", "Hitam",
    "Selamat pagi", "Selamat petang", "Terima kasih", "Sama-sama", "Idle"
]

# Signs with low hand detection in training (need augmentation with hand visibility)
LOW_HAND_DETECTION_SIGNS = {
    "Biru": 0.23,
    "Hitam": 0.24,
    "Hijau": 0.29,
    "C": 0.35,
    "B": 0.43,
    "D": 0.48,
}


def load_data():
    """Load training data with enhanced processing."""
    print("\n📂 Loading data...")
    
    actions = sorted([f.name for f in DATA_PATH.iterdir() if f.is_dir()])
    if not actions:
        print("✗ No data found!")
        return None, None, None
    
    print(f"   Actions: {actions}")
    
    label_map = {action: idx for idx, action in enumerate(actions)}
    sequences = []
    labels = []
    metadata = []  # Store metadata for each sequence
    
    for action in actions:
        action_path = DATA_PATH / action
        seq_folders = sorted(
            [f for f in action_path.iterdir() if f.is_dir() and f.name.isdigit()],
            key=lambda f: int(f.name)
        )
        
        count = 0
        for seq_folder in seq_folders:
            frames = []
            hand_detected_frames = 0
            valid = True
            
            for frame_num in range(1, SEQUENCE_LENGTH + 1):
                frame_file = seq_folder / f"{frame_num}.npy"
                if frame_file.exists():
                    try:
                        data = np.load(str(frame_file))
                        if data.shape[0] == NUM_FEATURES:
                            frames.append(data)
                            # Check hand detection
                            if np.any(data[RIGHT_HAND_START:RIGHT_HAND_END] != 0):
                                hand_detected_frames += 1
                        else:
                            valid = False
                            break
                    except:
                        frames.append(np.zeros(NUM_FEATURES))
                else:
                    frames.append(np.zeros(NUM_FEATURES))
            
            if valid and len(frames) == SEQUENCE_LENGTH:
                sequences.append(np.array(frames))
                labels.append(label_map[action])
                metadata.append({
                    "action": action,
                    "hand_detection_rate": hand_detected_frames / SEQUENCE_LENGTH,
                    "is_problem_sign": action in PROBLEM_SIGNS
                })
                count += 1
        
        print(f"   {action}: {count} sequences")
    
    return np.array(sequences), np.array(labels), np.array(actions), metadata


def compute_hand_features(sequence):
    """Extract additional hand-specific features for better discrimination."""
    features = []
    
    for frame in sequence:
        # Right hand position relative to face (nose at pose index 0)
        nose = frame[0:3]
        right_wrist = frame[RIGHT_HAND_START:RIGHT_HAND_START+3]
        left_wrist = frame[LEFT_HAND_START:LEFT_HAND_START+3]
        
        # Hand-to-face distances
        rh_to_nose = np.linalg.norm(right_wrist - nose) if np.any(right_wrist != 0) else 0
        lh_to_nose = np.linalg.norm(left_wrist - nose) if np.any(left_wrist != 0) else 0
        
        # Hand detected flags
        rh_detected = 1.0 if np.any(frame[RIGHT_HAND_START:RIGHT_HAND_END] != 0) else 0.0
        lh_detected = 1.0 if np.any(frame[LEFT_HAND_START:LEFT_HAND_END] != 0) else 0.0
        
        # Finger spread (distance between thumb tip and pinky tip)
        if rh_detected:
            thumb_tip = frame[RIGHT_HAND_START + 4*3:RIGHT_HAND_START + 4*3 + 3]
            pinky_tip = frame[RIGHT_HAND_START + 20*3:RIGHT_HAND_START + 20*3 + 3]
            rh_spread = np.linalg.norm(thumb_tip - pinky_tip)
        else:
            rh_spread = 0
        
        features.append([rh_to_nose, lh_to_nose, rh_detected, lh_detected, rh_spread])
    
    return np.array(features)


def augment_sequence_enhanced(sequence, action, preserve_structure=True):
    """Enhanced augmentation that preserves discriminative features."""
    aug = sequence.copy()
    
    # Different augmentation strategies based on sign type
    is_problem = action in PROBLEM_SIGNS
    has_low_hand_detection = action in LOW_HAND_DETECTION_SIGNS
    
    # 1. Spatial augmentation (smaller for problem signs to preserve features)
    scale = np.random.uniform(0.95, 1.05) if is_problem else np.random.uniform(0.9, 1.1)
    
    # 2. Position jitter (smaller for greetings which rely on motion)
    if action.startswith("Selamat") or action in ["Terima kasih", "Sama-sama"]:
        jitter = np.random.uniform(-0.01, 0.01, (SEQUENCE_LENGTH, NUM_FEATURES))
    else:
        jitter = np.random.uniform(-0.02, 0.02, (SEQUENCE_LENGTH, NUM_FEATURES))
    
    # Apply augmentation
    for i in range(SEQUENCE_LENGTH):
        # Scale around center
        pose_data = aug[i, POSE_START:POSE_END].reshape(-1, 4)
        pose_data[:, :3] = pose_data[:, :3] * scale
        aug[i, POSE_START:POSE_END] = pose_data.flatten()
        
        # Scale hands
        for hand_start, hand_end in [(LEFT_HAND_START, LEFT_HAND_END), (RIGHT_HAND_START, RIGHT_HAND_END)]:
            if np.any(aug[i, hand_start:hand_end] != 0):
                hand_data = aug[i, hand_start:hand_end].reshape(-1, 3)
                hand_data = hand_data * scale
                aug[i, hand_start:hand_end] = hand_data.flatten()
    
    # Add jitter
    aug = aug + jitter
    
    # 3. Temporal augmentation (only for non-problem signs)
    if not is_problem and random.random() < 0.3:
        # Small temporal shift
        shift = random.randint(-2, 2)
        if shift != 0:
            aug = np.roll(aug, shift, axis=0)
    
    return aug


def create_confusion_aware_pairs(X, y, actions, metadata):
    """Create training pairs that emphasize confused sign pairs."""
    pairs = []
    pair_labels = []  # 1 = same class, 0 = different class
    
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    
    # Group sequences by action
    action_sequences = {action: [] for action in actions}
    for i, (seq, label) in enumerate(zip(X, y)):
        action_sequences[actions[label]].append(seq)
    
    # Create positive pairs (same class)
    for action in actions:
        seqs = action_sequences[action]
        for i in range(len(seqs)):
            for j in range(i+1, min(i+3, len(seqs))):  # Limit pairs per sequence
                pairs.append((seqs[i], seqs[j]))
                pair_labels.append(1)
    
    # Create hard negative pairs (confused classes)
    for action1, action2 in CONFUSION_PAIRS:
        if action1 in action_to_idx and action2 in action_to_idx:
            seqs1 = action_sequences[action1]
            seqs2 = action_sequences[action2]
            
            # Create negative pairs between confused classes
            for s1 in seqs1[:5]:  # Limit to avoid too many pairs
                for s2 in seqs2[:5]:
                    pairs.append((s1, s2))
                    pair_labels.append(0)
    
    return pairs, pair_labels


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance and hard examples."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Convert y_true to one-hot encoding
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
        
        # Squeeze if needed (from shape [batch, 1, classes] to [batch, classes])
        if len(y_true_one_hot.shape) == 3:
            y_true_one_hot = tf.squeeze(y_true_one_hot, axis=1)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    return focal_loss_fn


def build_optimized_model(num_classes, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)):
    """Build model with architecture optimized for discriminating similar signs."""
    
    inputs = keras.layers.Input(shape=input_shape, name='input')
    
    # 1. Feature extraction branch for pose
    pose_slice = keras.layers.Lambda(lambda x: x[:, :, POSE_START:POSE_END])(inputs)
    pose_features = keras.layers.Dense(64, activation='relu')(pose_slice)
    pose_features = keras.layers.Dropout(0.2)(pose_features)
    
    # 2. Feature extraction branch for hands (critical for sign distinction)
    hand_slice = keras.layers.Lambda(lambda x: x[:, :, LEFT_HAND_START:RIGHT_HAND_END])(inputs)
    hand_features = keras.layers.Dense(96, activation='relu')(hand_slice)
    hand_features = keras.layers.Dropout(0.2)(hand_features)
    
    # 3. Combine features
    combined = keras.layers.Concatenate()([pose_features, hand_features])
    
    # 4. Temporal processing with attention-like mechanism
    # First LSTM layer
    x = keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(combined)
    
    # Second LSTM layer with skip connection preparation
    lstm_out = keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    
    # 5. Temporal attention - focus on key frames
    attention_weights = keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention_weights = keras.layers.Softmax(axis=1)(attention_weights)
    attended = keras.layers.Multiply()([lstm_out, attention_weights])
    
    # 6. Final LSTM
    x = keras.layers.LSTM(64, return_sequences=False, dropout=0.3)(attended)
    
    # 7. Classification head with increased capacity
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def compute_class_weights_from_confusion(y, actions, confusion_multiplier=2.0):
    """Compute class weights based on confusion patterns."""
    # Base class weights from sklearn
    classes = np.unique(y)
    base_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {i: w for i, w in enumerate(base_weights)}
    
    # Increase weights for problem signs
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    for sign in PROBLEM_SIGNS:
        if sign in action_to_idx:
            idx = action_to_idx[sign]
            class_weights[idx] *= confusion_multiplier
            print(f"   Increased weight for {sign}: {class_weights[idx]:.2f}")
    
    return class_weights


def augment_dataset(X, y, actions, metadata, aug_factor=5):
    """Augment dataset with focus on problem signs."""
    X_aug = list(X)
    y_aug = list(y)
    
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    
    for i, (seq, label) in enumerate(zip(X, y)):
        action = actions[label]
        
        # More augmentation for problem signs
        factor = aug_factor * 2 if action in PROBLEM_SIGNS else aug_factor
        
        for _ in range(factor):
            aug_seq = augment_sequence_enhanced(seq, action)
            X_aug.append(aug_seq)
            y_aug.append(label)
    
    return np.array(X_aug), np.array(y_aug)


class ConfusionAwareCallback(keras.callbacks.Callback):
    """Custom callback to monitor confusion between problem pairs."""
    
    def __init__(self, X_val, y_val, actions):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.actions = actions
        self.action_to_idx = {action: idx for idx, action in enumerate(actions)}
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            predictions = self.model.predict(self.X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            print(f"\n   📊 Confusion check at epoch {epoch + 1}:")
            
            for sign1, sign2 in CONFUSION_PAIRS[:5]:  # Check top 5 pairs
                if sign1 in self.action_to_idx and sign2 in self.action_to_idx:
                    idx1 = self.action_to_idx[sign1]
                    idx2 = self.action_to_idx[sign2]
                    
                    # Check how often sign1 is predicted as sign2
                    mask1 = self.y_val == idx1
                    if np.sum(mask1) > 0:
                        confused = np.sum(pred_classes[mask1] == idx2)
                        total = np.sum(mask1)
                        rate = confused / total if total > 0 else 0
                        status = "✓" if rate < 0.1 else "⚠️" if rate < 0.3 else "✗"
                        print(f"      {status} {sign1}→{sign2}: {rate:.0%}")


def train_optimized_model():
    """Main training function with all optimizations."""
    print("\n" + "="*60)
    print("OPTIMIZED MODEL TRAINING")
    print("="*60)
    print("\nFocusing on problem signs:")
    for sign in PROBLEM_SIGNS:
        print(f"   - {sign}")
    
    # Load data
    X, y, actions, metadata = load_data()
    if X is None:
        return
    
    print(f"\n📊 Original data: {len(X)} sequences, {len(actions)} classes")
    
    # Augment with focus on problem signs
    print("\n🔄 Augmenting data (extra for problem signs)...")
    X_aug, y_aug = augment_dataset(X, y, actions, metadata, aug_factor=3)
    print(f"   Augmented data: {len(X_aug)} sequences")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Compute class weights
    print("\n⚖️  Computing class weights...")
    class_weights = compute_class_weights_from_confusion(y_train, actions)
    
    # Build model
    print("\n🏗️  Building optimized model...")
    model = build_optimized_model(len(actions))
    model.summary()
    
    # Compile with focal loss
    print("\n⚙️  Compiling with focal loss...")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / 'best_model_optimized.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ConfusionAwareCallback(X_val, y_val, actions)
    ]
    
    # Train
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    print("\n📊 Per-class accuracy:")
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    
    for action in actions:
        idx = action_to_idx[action]
        mask = y_val == idx
        if np.sum(mask) > 0:
            correct = np.sum(pred_classes[mask] == idx)
            total = np.sum(mask)
            acc = correct / total
            is_problem = action in PROBLEM_SIGNS
            marker = "🔴" if is_problem else "  "
            status = "✓" if acc >= 0.8 else "~" if acc >= 0.6 else "✗"
            print(f"   {marker} {status} {action}: {acc:.0%} ({correct}/{total})")
    
    # Check confusion pairs
    print("\n📊 Confusion pair analysis:")
    for sign1, sign2 in CONFUSION_PAIRS:
        if sign1 in action_to_idx and sign2 in action_to_idx:
            idx1 = action_to_idx[sign1]
            idx2 = action_to_idx[sign2]
            
            mask1 = y_val == idx1
            if np.sum(mask1) > 0:
                confused = np.sum(pred_classes[mask1] == idx2)
                total = np.sum(mask1)
                rate = confused / total if total > 0 else 0
                status = "✓" if rate < 0.1 else "⚠️" if rate < 0.3 else "✗"
                print(f"   {status} {sign1} mistaken as {sign2}: {rate:.0%}")
    
    # Save model
    print("\n💾 Saving models...")
    model.save(str(OUTPUT_DIR / 'action_model_optimized.keras'))
    
    # Save action labels
    np.save(str(OUTPUT_DIR / 'action_labels.npy'), actions)
    with open(str(OUTPUT_DIR / 'action_labels.txt'), 'w', encoding='utf-8') as f:
        for action in actions:
            f.write(action + '\n')
    
    # Convert to TFLite
    print("\n📱 Converting to TFLite...")
    convert_to_tflite(model, actions)
    
    print("\n✅ Training complete!")
    print(f"   Models saved to: {OUTPUT_DIR}")
    
    return model, history


def convert_to_tflite(model, actions):
    """Convert model to TFLite format."""
    # Standard conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    output_path = OUTPUT_DIR / 'action_model_optimized.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"   Saved: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
    
    # Save metadata
    metadata = {
        "model_name": "action_model_optimized",
        "input_shape": [1, SEQUENCE_LENGTH, NUM_FEATURES],
        "output_shape": [1, len(actions)],
        "actions": list(actions),
        "optimized_for": PROBLEM_SIGNS,
        "confusion_pairs_addressed": CONFUSION_PAIRS,
        "created": datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / 'model_metadata_optimized.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    model, history = train_optimized_model()
