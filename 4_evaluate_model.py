"""
Model Evaluation Script
=======================
This script evaluates your trained model to check:
1. Overall accuracy
2. Per-class accuracy (to detect bias)
3. Confusion matrix (which signs get confused with each other)
4. Class distribution in training data

Run this BEFORE deploying to mobile to ensure model quality.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import json

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
DATA_PATH = Path("C:/fyp/SL_Data_Processed")
SEQUENCE_LENGTH = 30
NUM_FEATURES = 258
# ---------------------

def load_from_processed_data(data_path):
    """Load data from SL_Data_Processed folder structure (same as training)."""
    sequences = []
    labels = []
    
    actions = sorted([f.name for f in data_path.iterdir() if f.is_dir()])
    if not actions:
        return None, None, None
    
    label_map = {action: idx for idx, action in enumerate(actions)}
    
    for action in actions:
        action_path = data_path / action
        seq_folders = sorted(
            [f for f in action_path.iterdir() if f.is_dir() and f.name.isdigit()],
            key=lambda f: int(f.name)
        )
        
        for seq_folder in seq_folders:
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
                sequences.append(np.array(frames))
                labels.append(label_map[action])
    
    if not sequences:
        return None, None, None
    
    return np.array(sequences), np.array(labels), np.array(actions)

def load_data():
    """Load the training/test data."""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    X, y, labels = None, None, None
    
    # Try to load from SL_Data_Processed folder first (same as training script)
    if DATA_PATH.exists():
        print(f"Loading from {DATA_PATH}...")
        X, y, labels = load_from_processed_data(DATA_PATH)
        if X is not None:
            print(f"✓ Loaded data from {DATA_PATH}")
    
    # Fallback: Try to find data files in current directory
    if X is None:
        data_files = [
            ('sequences.npy', 'labels.npy'),
            ('X_train.npy', 'y_train.npy'),
        ]
        
        for x_file, y_file in data_files:
            if Path(x_file).exists() and Path(y_file).exists():
                X = np.load(x_file)
                y = np.load(y_file)
                print(f"✓ Loaded data from {x_file} and {y_file}")
                break
    
    # Fallback: Try MP_Data folder
    if X is None:
        mp_data = Path('MP_Data')
        if mp_data.exists():
            print("Loading from MP_Data folder...")
            X, y = load_from_mp_data(mp_data)
    
    if X is None:
        print("✗ No data files found!")
        print(f"  Expected: {DATA_PATH}")
        print("  Or: sequences.npy + labels.npy")
        print("  Or: X_train.npy + y_train.npy")
        print("  Or: MP_Data folder")
        return None, None, None
    
    # Load labels if not already loaded
    if labels is None:
        if Path('action_labels.npy').exists():
            labels = np.load('action_labels.npy')
        elif Path('action_labels.txt').exists():
            with open('action_labels.txt', 'r') as f:
                labels = np.array([line.strip() for line in f if line.strip()])
    
    print(f"  Data shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Classes: {labels}")
    
    return X, y, labels

def load_from_mp_data(mp_data_path):
    """Load data from MP_Data folder structure."""
    sequences = []
    labels = []
    
    actions = sorted([d.name for d in mp_data_path.iterdir() if d.is_dir()])
    
    for action_idx, action in enumerate(actions):
        action_path = mp_data_path / action
        for seq_folder in sorted(action_path.iterdir()):
            if seq_folder.is_dir():
                frames = []
                for frame_num in range(30):  # Assuming 30 frames
                    frame_path = seq_folder / f"{frame_num}.npy"
                    if frame_path.exists():
                        frames.append(np.load(str(frame_path)))
                
                if len(frames) == 30:
                    sequences.append(frames)
                    labels.append(action_idx)
    
    return np.array(sequences), np.array(labels)

def evaluate_keras_model(model_path, X, y, labels):
    """Evaluate a Keras model."""
    print("\n" + "="*60)
    print("EVALUATING KERAS MODEL")
    print("="*60)
    
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # If y is one-hot encoded, convert to class indices
    if len(y.shape) > 1:
        true_classes = np.argmax(y, axis=1)
    else:
        true_classes = y
    
    return predicted_classes, true_classes, predictions

def evaluate_tflite_model(model_path, X, y, labels):
    """Evaluate a TFLite model."""
    print("\n" + "="*60)
    print("EVALUATING TFLITE MODEL")
    print("="*60)
    
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    predictions = []
    for i, sample in enumerate(X):
        input_data = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0])
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X)} samples...")
    
    predictions = np.array(predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    
    if len(y.shape) > 1:
        true_classes = np.argmax(y, axis=1)
    else:
        true_classes = y
    
    return predicted_classes, true_classes, predictions

def calculate_metrics(predicted_classes, true_classes, predictions, labels):
    """Calculate and display all metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    num_classes = len(labels)
    
    # Overall accuracy
    correct = np.sum(predicted_classes == true_classes)
    total = len(true_classes)
    overall_accuracy = correct / total
    
    print(f"\n📊 OVERALL ACCURACY: {overall_accuracy:.1%} ({correct}/{total})")
    
    # Per-class metrics
    print(f"\n📊 PER-CLASS ACCURACY:")
    print("-" * 50)
    print(f"{'Class':<15} {'Accuracy':<12} {'Samples':<10} {'Correct':<10}")
    print("-" * 50)
    
    class_accuracies = []
    class_counts = []
    
    for i, label in enumerate(labels):
        mask = true_classes == i
        class_total = np.sum(mask)
        class_correct = np.sum((predicted_classes == true_classes) & mask)
        
        if class_total > 0:
            class_acc = class_correct / class_total
        else:
            class_acc = 0
        
        class_accuracies.append(class_acc)
        class_counts.append(class_total)
        
        # Color coding for accuracy
        if class_acc >= 0.9:
            status = "✓"
        elif class_acc >= 0.7:
            status = "~"
        else:
            status = "✗"
        
        print(f"{status} {label:<13} {class_acc:>8.1%}     {class_total:<10} {class_correct:<10}")
    
    print("-" * 50)
    
    # Check for bias
    print(f"\n📊 BIAS ANALYSIS:")
    min_acc = min(class_accuracies)
    max_acc = max(class_accuracies)
    acc_range = max_acc - min_acc
    
    min_class = labels[np.argmin(class_accuracies)]
    max_class = labels[np.argmax(class_accuracies)]
    
    print(f"  Best class:  {max_class} ({max_acc:.1%})")
    print(f"  Worst class: {min_class} ({min_acc:.1%})")
    print(f"  Accuracy range: {acc_range:.1%}")
    
    if acc_range > 0.3:
        print(f"  ⚠️  HIGH BIAS DETECTED! Some classes perform much worse than others.")
    elif acc_range > 0.15:
        print(f"  ⚠️  Moderate bias detected. Consider collecting more data for weak classes.")
    else:
        print(f"  ✓ Low bias - model performs consistently across classes.")
    
    # Data balance check
    print(f"\n📊 DATA BALANCE:")
    min_samples = min(class_counts)
    max_samples = max(class_counts)
    
    print(f"  Min samples: {min_samples} ({labels[np.argmin(class_counts)]})")
    print(f"  Max samples: {max_samples} ({labels[np.argmax(class_counts)]})")
    
    if max_samples > min_samples * 2:
        print(f"  ⚠️  IMBALANCED DATA! Some classes have much more data than others.")
    else:
        print(f"  ✓ Data is reasonably balanced.")
    
    # Confusion matrix
    print(f"\n📊 CONFUSION MATRIX:")
    print("(Rows = True class, Columns = Predicted class)")
    print()
    
    # Header
    header = "        " + " ".join([f"{l[:5]:>5}" for l in labels])
    print(header)
    print("-" * len(header))
    
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_classes, predicted_classes):
        confusion[true][pred] += 1
    
    for i, label in enumerate(labels):
        row = f"{label[:6]:<6} |"
        for j in range(num_classes):
            if i == j:
                row += f" {confusion[i][j]:>4}*"  # Correct predictions marked with *
            elif confusion[i][j] > 0:
                row += f" {confusion[i][j]:>4}!"  # Misclassifications marked with !
            else:
                row += f" {confusion[i][j]:>4} "
        print(row)
    
    # Most common confusions
    print(f"\n📊 MOST COMMON CONFUSIONS:")
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i][j] > 0:
                confusions.append((labels[i], labels[j], confusion[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    if confusions:
        for true_label, pred_label, count in confusions[:5]:
            print(f"  '{true_label}' mistaken as '{pred_label}': {count} times")
    else:
        print("  No confusions found!")
    
    # Confidence analysis
    print(f"\n📊 CONFIDENCE ANALYSIS:")
    correct_mask = predicted_classes == true_classes
    incorrect_mask = ~correct_mask
    
    correct_confidences = np.max(predictions[correct_mask], axis=1)
    incorrect_confidences = np.max(predictions[incorrect_mask], axis=1) if np.any(incorrect_mask) else np.array([])
    
    print(f"  Average confidence (correct):   {np.mean(correct_confidences):.1%}")
    if len(incorrect_confidences) > 0:
        print(f"  Average confidence (incorrect): {np.mean(incorrect_confidences):.1%}")
        
        # High confidence errors
        high_conf_errors = np.sum(incorrect_confidences > 0.7)
        print(f"  High-confidence errors (>70%):  {high_conf_errors}")
    
    return overall_accuracy, class_accuracies

def main():
    print("\n" + "="*60)
    print("MODEL EVALUATION TOOL")
    print("="*60)
    
    # Load data
    X, y, labels = load_data()
    if X is None:
        return
    
    # Find model
    keras_model = None
    tflite_model = None
    
    for f in ['action_model.h5', 'action_model.keras', 'model.h5', 'best_model.h5']:
        if Path(f).exists():
            keras_model = f
            break
    
    for f in ['action_model.tflite', 'model.tflite']:
        if Path(f).exists():
            tflite_model = f
            break
    
    if keras_model:
        print(f"\nFound Keras model: {keras_model}")
        try:
            pred, true, probs = evaluate_keras_model(keras_model, X, y, labels)
            calculate_metrics(pred, true, probs, labels)
        except Exception as e:
            print(f"Error evaluating Keras model: {e}")
    
    if tflite_model:
        print(f"\nFound TFLite model: {tflite_model}")
        try:
            pred, true, probs = evaluate_tflite_model(tflite_model, X, y, labels)
            calculate_metrics(pred, true, probs, labels)
        except Exception as e:
            print(f"Error evaluating TFLite model: {e}")
    
    if not keras_model and not tflite_model:
        print("\n✗ No model found!")
        print("  Expected: action_model.h5 or action_model.tflite")

if __name__ == "__main__":
    main()