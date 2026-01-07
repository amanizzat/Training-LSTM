"""
DIAGNOSTIC TEST SCRIPT
======================
Comprehensive testing tool to compare training data vs live predictions.
Tests each sign by category with 5 predictions per sign.
Shows what the AI "sees" during training vs live testing.

This helps diagnose why model shows 100% accuracy on training data
but performs poorly in live testing.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
import json
import time
from datetime import datetime

# --- Configuration ---
DATA_PATH = Path("C:/fyp/SL_Data_Processed")
MODEL_PATH = Path("C:/fyp/training/action_model.tflite")
OUTPUT_DIR = Path("C:/fyp/training/diagnostic_results")

SEQUENCE_LENGTH = 30
NUM_FEATURES = 258
TESTS_PER_SIGN = 5

# Sign categories
CATEGORIES = {
    "Numbers": ["1", "2", "3", "4", "5"],
    "Alphabet": ["A", "B", "C", "D", "E"],
    "Colors": ["Merah", "Biru", "Kuning", "Hijau", "Hitam"],
    "Greetings": ["Selamat pagi", "Selamat petang", "Selamat malam", "Terima kasih", "Sama-sama"],
    "Other": ["Idle"]
}

# Feature layout
POSE_START = 0
POSE_END = 132      # 33 landmarks × 4 values
LEFT_HAND_START = 132
LEFT_HAND_END = 195  # 21 landmarks × 3 values
RIGHT_HAND_START = 195
RIGHT_HAND_END = 258 # 21 landmarks × 3 values


class DiagnosticTester:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        self.actions = self._load_actions()
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        # Load training data statistics
        self.training_stats = self._compute_training_stats()
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "summary": {}
        }
        
        # Sequence buffer
        self.sequence = []
        
    def _load_actions(self):
        """Load action labels."""
        labels_file = Path("C:/fyp/training/action_labels.txt")
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            return sorted([f.name for f in DATA_PATH.iterdir() if f.is_dir()])
    
    def _compute_training_stats(self):
        """Compute statistics from training data for comparison."""
        print("\n📊 Computing training data statistics...")
        stats = {}
        
        for action in self.actions:
            action_path = DATA_PATH / action
            if not action_path.exists():
                continue
                
            all_frames = []
            seq_folders = sorted(
                [f for f in action_path.iterdir() if f.is_dir() and f.name.isdigit()],
                key=lambda f: int(f.name)
            )
            
            for seq_folder in seq_folders:
                for frame_num in range(1, SEQUENCE_LENGTH + 1):
                    frame_file = seq_folder / f"{frame_num}.npy"
                    if frame_file.exists():
                        data = np.load(str(frame_file))
                        if data.shape[0] == NUM_FEATURES:
                            all_frames.append(data)
            
            if all_frames:
                all_frames = np.array(all_frames)
                stats[action] = {
                    "mean": np.mean(all_frames, axis=0),
                    "std": np.std(all_frames, axis=0),
                    "min": np.min(all_frames, axis=0),
                    "max": np.max(all_frames, axis=0),
                    "num_sequences": len(seq_folders),
                    "num_frames": len(all_frames),
                    # Hand presence statistics
                    "left_hand_presence": np.mean(np.any(all_frames[:, LEFT_HAND_START:LEFT_HAND_END] != 0, axis=1)),
                    "right_hand_presence": np.mean(np.any(all_frames[:, RIGHT_HAND_START:RIGHT_HAND_END] != 0, axis=1)),
                    # Key landmark means
                    "right_wrist_mean": np.mean(all_frames[:, RIGHT_HAND_START:RIGHT_HAND_START+3], axis=0),
                    "left_wrist_mean": np.mean(all_frames[:, LEFT_HAND_START:LEFT_HAND_START+3], axis=0),
                }
                print(f"   {action}: {len(seq_folders)} sequences, {len(all_frames)} frames")
        
        return stats
    
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results (same as training)."""
        # Pose (33 landmarks × 4 values = 132)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # Left hand (21 landmarks × 3 values = 63)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right hand (21 landmarks × 3 values = 63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, lh, rh])
    
    def analyze_frame_diff(self, live_frame, action):
        """Analyze difference between live frame and training data."""
        if action not in self.training_stats:
            return None
        
        train_stats = self.training_stats[action]
        
        # Compute z-score (how many std deviations from training mean)
        diff = live_frame - train_stats["mean"]
        std = train_stats["std"]
        std[std == 0] = 1  # Avoid division by zero
        z_score = diff / std
        
        # Hand presence
        live_left_hand = np.any(live_frame[LEFT_HAND_START:LEFT_HAND_END] != 0)
        live_right_hand = np.any(live_frame[RIGHT_HAND_START:RIGHT_HAND_END] != 0)
        
        return {
            "mean_z_score": np.mean(np.abs(z_score)),
            "max_z_score": np.max(np.abs(z_score)),
            "live_left_hand": live_left_hand,
            "live_right_hand": live_right_hand,
            "train_left_hand_rate": train_stats["left_hand_presence"],
            "train_right_hand_rate": train_stats["right_hand_presence"],
            "pose_diff": np.mean(np.abs(z_score[POSE_START:POSE_END])),
            "left_hand_diff": np.mean(np.abs(z_score[LEFT_HAND_START:LEFT_HAND_END])),
            "right_hand_diff": np.mean(np.abs(z_score[RIGHT_HAND_START:RIGHT_HAND_END])),
        }
    
    def predict(self, sequence):
        """Run prediction on sequence."""
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]
    
    def run_single_test(self, expected_action, cap, show_analysis=True):
        """Run a single test for expected action."""
        self.sequence = []
        predictions_during_capture = []
        frame_analyses = []
        
        print(f"\n   ⏳ Capturing {SEQUENCE_LENGTH} frames...")
        
        start_time = time.time()
        frames_captured = 0
        
        while frames_captured < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract keypoints
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            frames_captured += 1
            
            # Analyze this frame vs training data
            analysis = self.analyze_frame_diff(keypoints, expected_action)
            if analysis:
                frame_analyses.append(analysis)
            
            # Draw landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
            # Show progress
            progress = f"Capturing: {frames_captured}/{SEQUENCE_LENGTH}"
            cv2.putText(image, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Expected: {expected_action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Show hand detection status
            lh_status = "LEFT HAND: " + ("DETECTED" if results.left_hand_landmarks else "NOT DETECTED")
            rh_status = "RIGHT HAND: " + ("DETECTED" if results.right_hand_landmarks else "NOT DETECTED")
            lh_color = (0, 255, 0) if results.left_hand_landmarks else (0, 0, 255)
            rh_color = (0, 255, 0) if results.right_hand_landmarks else (0, 0, 255)
            cv2.putText(image, lh_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lh_color, 2)
            cv2.putText(image, rh_status, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rh_color, 2)
            
            cv2.imshow('Diagnostic Test', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        
        capture_time = time.time() - start_time
        
        # Make prediction
        sequence_array = np.array(self.sequence)
        probabilities = self.predict(sequence_array)
        predicted_idx = np.argmax(probabilities)
        predicted_action = self.actions[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [(self.actions[i], probabilities[i]) for i in top3_idx]
        
        # Aggregate frame analyses
        avg_analysis = {}
        if frame_analyses:
            avg_analysis = {
                "mean_z_score": np.mean([a["mean_z_score"] for a in frame_analyses]),
                "max_z_score": np.max([a["max_z_score"] for a in frame_analyses]),
                "live_left_hand_rate": np.mean([a["live_left_hand"] for a in frame_analyses]),
                "live_right_hand_rate": np.mean([a["live_right_hand"] for a in frame_analyses]),
                "train_left_hand_rate": frame_analyses[0]["train_left_hand_rate"],
                "train_right_hand_rate": frame_analyses[0]["train_right_hand_rate"],
                "pose_diff": np.mean([a["pose_diff"] for a in frame_analyses]),
                "left_hand_diff": np.mean([a["left_hand_diff"] for a in frame_analyses]),
                "right_hand_diff": np.mean([a["right_hand_diff"] for a in frame_analyses]),
            }
        
        result = {
            "expected": expected_action,
            "predicted": predicted_action,
            "correct": predicted_action == expected_action,
            "confidence": float(confidence),
            "top3": [(a, float(p)) for a, p in top3],
            "all_probabilities": {self.actions[i]: float(probabilities[i]) for i in range(len(self.actions))},
            "capture_time": capture_time,
            "analysis": avg_analysis,
            "sequence_stats": {
                "left_hand_detected_frames": int(np.sum([np.any(f[LEFT_HAND_START:LEFT_HAND_END] != 0) for f in self.sequence])),
                "right_hand_detected_frames": int(np.sum([np.any(f[RIGHT_HAND_START:RIGHT_HAND_END] != 0) for f in self.sequence])),
                "total_frames": SEQUENCE_LENGTH
            }
        }
        
        # Print result
        status = "✓" if result["correct"] else "✗"
        print(f"   {status} Expected: {expected_action} | Predicted: {predicted_action} ({confidence:.1%})")
        print(f"      Top 3: {', '.join([f'{a}({p:.1%})' for a, p in top3])}")
        
        if show_analysis and avg_analysis:
            print(f"      📊 Analysis vs Training Data:")
            print(f"         Mean Z-Score: {avg_analysis['mean_z_score']:.2f} (lower=more similar to training)")
            print(f"         Live Left Hand: {avg_analysis['live_left_hand_rate']:.0%} | Training: {avg_analysis['train_left_hand_rate']:.0%}")
            print(f"         Live Right Hand: {avg_analysis['live_right_hand_rate']:.0%} | Training: {avg_analysis['train_right_hand_rate']:.0%}")
            print(f"         Pose Diff: {avg_analysis['pose_diff']:.2f} | LH Diff: {avg_analysis['left_hand_diff']:.2f} | RH Diff: {avg_analysis['right_hand_diff']:.2f}")
        
        return result
    
    def test_category(self, category_name, signs, cap):
        """Test all signs in a category."""
        print(f"\n{'='*60}")
        print(f"TESTING CATEGORY: {category_name}")
        print(f"{'='*60}")
        
        category_results = {}
        
        for sign in signs:
            if sign not in self.actions:
                print(f"\n⚠️  Sign '{sign}' not found in model. Skipping...")
                continue
            
            print(f"\n📝 Testing sign: {sign}")
            print("-" * 40)
            
            sign_results = []
            
            for test_num in range(1, TESTS_PER_SIGN + 1):
                # Wait for user to be ready
                print(f"\n   Test {test_num}/{TESTS_PER_SIGN}")
                self._show_ready_screen(cap, sign, test_num)
                
                # Run test
                result = self.run_single_test(sign, cap)
                if result is None:
                    print("   Test cancelled")
                    continue
                
                sign_results.append(result)
                
                # Brief pause between tests
                time.sleep(0.5)
            
            # Compute sign statistics
            if sign_results:
                correct_count = sum(1 for r in sign_results if r["correct"])
                avg_confidence = np.mean([r["confidence"] for r in sign_results])
                
                # Most common wrong predictions
                wrong_predictions = [r["predicted"] for r in sign_results if not r["correct"]]
                
                category_results[sign] = {
                    "tests": sign_results,
                    "accuracy": correct_count / len(sign_results),
                    "correct_count": correct_count,
                    "total_tests": len(sign_results),
                    "avg_confidence": float(avg_confidence),
                    "wrong_predictions": wrong_predictions
                }
                
                print(f"\n   📊 Sign '{sign}' Summary: {correct_count}/{len(sign_results)} correct ({correct_count/len(sign_results):.0%})")
                if wrong_predictions:
                    print(f"      Wrong predictions: {wrong_predictions}")
        
        return category_results
    
    def _show_ready_screen(self, cap, sign, test_num):
        """Show ready screen and wait for spacebar."""
        print(f"   Press SPACEBAR when ready to perform '{sign}'...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Draw instructions
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
            cv2.putText(frame, f"Test {test_num}/{TESTS_PER_SIGN}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Sign: {sign}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, "Press SPACEBAR when ready", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            cv2.imshow('Diagnostic Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Countdown
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
                    cv2.putText(frame, f"Starting in {i}...", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow('Diagnostic Test', frame)
                    cv2.waitKey(1000)
                return
            elif key == ord('q'):
                return None
    
    def run_full_diagnostic(self):
        """Run full diagnostic test for all categories."""
        print("\n" + "="*60)
        print("DIAGNOSTIC TEST - TRAINING vs LIVE COMPARISON")
        print("="*60)
        print(f"Model: {MODEL_PATH}")
        print(f"Training Data: {DATA_PATH}")
        print(f"Tests per sign: {TESTS_PER_SIGN}")
        print(f"Actions: {self.actions}")
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            # Show initial instructions
            self._show_instructions(cap)
            
            # Test each category
            for category_name, signs in CATEGORIES.items():
                # Check if user wants to test this category
                if not self._ask_category(cap, category_name, signs):
                    continue
                
                category_results = self.test_category(category_name, signs, cap)
                self.results["categories"][category_name] = category_results
            
            # Compute overall summary
            self._compute_summary()
            
            # Save results
            self._save_results()
            
            # Show final summary
            self._show_final_summary(cap)
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _show_instructions(self, cap):
        """Show initial instructions."""
        print("\n📋 Instructions:")
        print("   1. Each sign will be tested 5 times")
        print("   2. Press SPACEBAR when ready to perform a sign")
        print("   3. Hold the sign steady during capture")
        print("   4. Press Q to skip or quit")
        print("\nPress SPACEBAR to begin...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 200), (0, 0, 0), -1)
            cv2.putText(frame, "DIAGNOSTIC TEST", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, "This will test each sign 5 times", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "and compare with training data", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACEBAR to begin", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Diagnostic Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                return
            elif key == ord('q'):
                return None
    
    def _ask_category(self, cap, category_name, signs):
        """Ask if user wants to test this category."""
        print(f"\n📁 Category: {category_name}")
        print(f"   Signs: {signs}")
        print("   Press SPACEBAR to test, S to skip, Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 180), (0, 0, 0), -1)
            cv2.putText(frame, f"Category: {category_name}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            y = 80
            for i, sign in enumerate(signs[:5]):
                cv2.putText(frame, sign, (10 + (i % 3) * 150, y + (i // 3) * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, "SPACE=Test | S=Skip | Q=Quit", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Diagnostic Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                return True
            elif key == ord('s'):
                print("   Skipping category...")
                return False
            elif key == ord('q'):
                return False
    
    def _compute_summary(self):
        """Compute overall summary statistics."""
        total_tests = 0
        total_correct = 0
        category_accuracies = {}
        problem_signs = []
        
        for category_name, category_results in self.results["categories"].items():
            cat_tests = 0
            cat_correct = 0
            
            for sign, sign_data in category_results.items():
                total_tests += sign_data["total_tests"]
                total_correct += sign_data["correct_count"]
                cat_tests += sign_data["total_tests"]
                cat_correct += sign_data["correct_count"]
                
                # Track problem signs (< 60% accuracy)
                if sign_data["accuracy"] < 0.6:
                    problem_signs.append({
                        "sign": sign,
                        "category": category_name,
                        "accuracy": sign_data["accuracy"],
                        "wrong_predictions": sign_data["wrong_predictions"]
                    })
            
            if cat_tests > 0:
                category_accuracies[category_name] = cat_correct / cat_tests
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total_tests if total_tests > 0 else 0,
            "category_accuracies": category_accuracies,
            "problem_signs": problem_signs
        }
    
    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"diagnostic_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Also save a summary text file
        summary_file = OUTPUT_DIR / f"diagnostic_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DIAGNOSTIC TEST SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            summary = self.results["summary"]
            f.write(f"Overall Accuracy: {summary['overall_accuracy']:.1%} ({summary['total_correct']}/{summary['total_tests']})\n\n")
            
            f.write("Category Accuracies:\n")
            for cat, acc in summary["category_accuracies"].items():
                f.write(f"  {cat}: {acc:.1%}\n")
            
            f.write("\nProblem Signs (< 60% accuracy):\n")
            if summary["problem_signs"]:
                for ps in summary["problem_signs"]:
                    f.write(f"  {ps['sign']} ({ps['category']}): {ps['accuracy']:.0%}\n")
                    f.write(f"    Confused with: {ps['wrong_predictions']}\n")
            else:
                f.write("  None\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for cat_name, cat_results in self.results["categories"].items():
                f.write(f"\n{cat_name}:\n")
                f.write("-"*40 + "\n")
                for sign, sign_data in cat_results.items():
                    f.write(f"  {sign}: {sign_data['correct_count']}/{sign_data['total_tests']} ({sign_data['accuracy']:.0%})\n")
                    for i, test in enumerate(sign_data['tests']):
                        status = "✓" if test['correct'] else "✗"
                        f.write(f"    Test {i+1}: {status} Predicted: {test['predicted']} ({test['confidence']:.0%})\n")
                        if test['analysis']:
                            f.write(f"           Z-Score: {test['analysis']['mean_z_score']:.2f}, ")
                            f.write(f"RH Live: {test['analysis']['live_right_hand_rate']:.0%} vs Train: {test['analysis']['train_right_hand_rate']:.0%}\n")
        
        print(f"💾 Summary saved to: {summary_file}")
    
    def _show_final_summary(self, cap):
        """Show final summary on screen."""
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"\n📊 Overall Accuracy: {summary['overall_accuracy']:.1%} ({summary['total_correct']}/{summary['total_tests']})")
        
        print("\n📊 Category Accuracies:")
        for cat, acc in summary["category_accuracies"].items():
            status = "✓" if acc >= 0.8 else ("~" if acc >= 0.6 else "✗")
            print(f"   {status} {cat}: {acc:.1%}")
        
        if summary["problem_signs"]:
            print("\n⚠️  Problem Signs (< 60% accuracy):")
            for ps in summary["problem_signs"]:
                print(f"   - {ps['sign']}: {ps['accuracy']:.0%}")
                print(f"     Often confused with: {ps['wrong_predictions']}")
        
        print("\n📁 Results saved to:", OUTPUT_DIR)
        print("\nPress any key to exit...")
        
        # Show on screen
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            
            y = 40
            cv2.putText(frame, "DIAGNOSTIC COMPLETE", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y += 50
            
            cv2.putText(frame, f"Overall: {summary['overall_accuracy']:.1%}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 40
            
            for cat, acc in list(summary["category_accuracies"].items())[:5]:
                color = (0, 255, 0) if acc >= 0.8 else ((0, 255, 255) if acc >= 0.6 else (0, 0, 255))
                cv2.putText(frame, f"{cat}: {acc:.1%}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y += 30
            
            cv2.putText(frame, "Press any key to exit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            cv2.imshow('Diagnostic Test', frame)
            
            if cv2.waitKey(1) & 0xFF != 255:
                break


def main():
    print("\n" + "="*60)
    print("SIGN LANGUAGE MODEL DIAGNOSTIC TEST")
    print("="*60)
    print("\nThis tool will:")
    print("  1. Test each sign by category (Numbers, Alphabet, Colors, Greetings)")
    print("  2. Run 5 tests per sign")
    print("  3. Compare live data with training data statistics")
    print("  4. Identify problem signs and confusion patterns")
    print("  5. Save detailed results for analysis")
    
    tester = DiagnosticTester()
    tester.run_full_diagnostic()


if __name__ == "__main__":
    main()
