"""
TARGETED DATA COLLECTION SCRIPT
================================
Collects additional training data for problem signs with real-time quality checking.

The diagnostic revealed that many problem signs have low hand detection in training:
- Biru: 23% hand detection (should be ~60%+)
- Hitam: 24%
- Hijau: 29%
- C: 35%
- B: 43%

This script:
1. Shows you exactly what the model saw during training
2. Ensures new data has proper hand detection
3. Validates data quality before saving
4. Focuses on distinguishing confused pairs

Usage:
    python 7_collect_problem_signs.py
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import time
from datetime import datetime

# --- Configuration ---
DATA_PATH = Path("C:/fyp/SL_Data_Processed")
SEQUENCE_LENGTH = 30
NUM_FEATURES = 258
MIN_HAND_DETECTION_RATE = 0.5  # Minimum 50% hand detection required

# Feature layout
POSE_START = 0
POSE_END = 132
LEFT_HAND_START = 132
LEFT_HAND_END = 195
RIGHT_HAND_START = 195
RIGHT_HAND_END = 258

# Problem signs that need more data
PROBLEM_SIGNS_TO_COLLECT = {
    "2": {"confused_with": ["3", "1"], "tip": "Show 2 fingers clearly spread apart"},
    "5": {"confused_with": ["3"], "tip": "Show all 5 fingers spread wide"},
    "B": {"confused_with": ["4", "E"], "tip": "Flat hand with thumb tucked"},
    "C": {"confused_with": ["D"], "tip": "Curved C shape, fingers together"},
    "E": {"confused_with": ["1"], "tip": "Curved fingers like claw"},
    "Biru": {"confused_with": ["D"], "tip": "Sign for blue color"},
    "Hijau": {"confused_with": ["Merah", "Kuning"], "tip": "Sign for green color"},
    "Hitam": {"confused_with": ["D"], "tip": "Sign for black color"},
    "Selamat petang": {"confused_with": ["Selamat malam"], "tip": "Good afternoon greeting"},
    "Terima kasih": {"confused_with": ["Merah"], "tip": "Thank you sign"},
    "Sama-sama": {"confused_with": ["Merah"], "tip": "You're welcome sign"},
    "Idle": {"confused_with": ["B", "2", "4"], "tip": "Hands down, relaxed position"},
}

SEQUENCES_PER_SIGN = 10  # Collect 10 new sequences per problem sign


class QualityDataCollector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load existing data statistics for comparison
        self.existing_stats = self._load_existing_stats()
    
    def _load_existing_stats(self):
        """Load statistics from existing training data."""
        stats = {}
        for sign in PROBLEM_SIGNS_TO_COLLECT.keys():
            sign_path = DATA_PATH / sign
            if sign_path.exists():
                all_frames = []
                seq_folders = sorted(
                    [f for f in sign_path.iterdir() if f.is_dir() and f.name.isdigit()],
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
                    rh_detection = np.mean(np.any(all_frames[:, RIGHT_HAND_START:RIGHT_HAND_END] != 0, axis=1))
                    lh_detection = np.mean(np.any(all_frames[:, LEFT_HAND_START:LEFT_HAND_END] != 0, axis=1))
                    stats[sign] = {
                        "num_sequences": len(seq_folders),
                        "rh_detection": rh_detection,
                        "lh_detection": lh_detection,
                        "mean": np.mean(all_frames, axis=0),
                        "std": np.std(all_frames, axis=0)
                    }
        return stats
    
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])
    
    def check_sequence_quality(self, frames, sign):
        """Check if a captured sequence meets quality requirements."""
        frames_array = np.array(frames)
        
        # Check hand detection rate
        rh_detected = np.sum([np.any(f[RIGHT_HAND_START:RIGHT_HAND_END] != 0) for f in frames]) / len(frames)
        lh_detected = np.sum([np.any(f[LEFT_HAND_START:LEFT_HAND_END] != 0) for f in frames]) / len(frames)
        
        # For Idle, we expect NO hands
        if sign == "Idle":
            if rh_detected > 0.1 or lh_detected > 0.1:
                return False, f"Idle should have no hands detected. RH: {rh_detected:.0%}, LH: {lh_detected:.0%}"
            return True, f"Good Idle capture (no hands: RH {rh_detected:.0%}, LH {lh_detected:.0%})"
        
        # For other signs, require hand detection
        if rh_detected < MIN_HAND_DETECTION_RATE and lh_detected < MIN_HAND_DETECTION_RATE:
            return False, f"Low hand detection: RH {rh_detected:.0%}, LH {lh_detected:.0%}. Need at least {MIN_HAND_DETECTION_RATE:.0%}"
        
        # Compare with existing training data
        if sign in self.existing_stats:
            existing = self.existing_stats[sign]
            # New data should have BETTER hand detection than existing
            if rh_detected < existing["rh_detection"] * 0.8:
                return False, f"Hand detection ({rh_detected:.0%}) worse than training data ({existing['rh_detection']:.0%})"
        
        return True, f"Good quality! RH: {rh_detected:.0%}, LH: {lh_detected:.0%}"
    
    def show_training_data_example(self, sign, cap):
        """Show what the training data looks like for comparison."""
        if sign not in self.existing_stats:
            return
        
        stats = self.existing_stats[sign]
        
        # Show statistics
        print(f"\n   📊 Existing training data for '{sign}':")
        print(f"      Sequences: {stats['num_sequences']}")
        print(f"      Right hand detection: {stats['rh_detection']:.0%}")
        print(f"      Left hand detection: {stats['lh_detection']:.0%}")
        
        if stats['rh_detection'] < 0.4:
            print(f"      ⚠️  WARNING: Low hand detection in training! This may be why it's confused.")
            print(f"      📌 TIP: Make sure your hand is clearly visible when recording new data.")
    
    def collect_sequence(self, sign, cap, seq_num, total_seqs):
        """Collect a single sequence for a sign."""
        frames = []
        
        print(f"\n   Collecting sequence {seq_num}/{total_seqs} for '{sign}'...")
        
        # Wait for user to be ready
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process to show landmarks
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
            # Show info
            info = PROBLEM_SIGNS_TO_COLLECT[sign]
            cv2.rectangle(image, (0, 0), (640, 160), (0, 0, 0), -1)
            cv2.putText(image, f"Sign: {sign} ({seq_num}/{total_seqs})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f"Tip: {info['tip']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"Often confused with: {', '.join(info['confused_with'])}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
            
            # Hand detection status
            lh_status = "LEFT: " + ("✓" if results.left_hand_landmarks else "✗")
            rh_status = "RIGHT: " + ("✓" if results.right_hand_landmarks else "✗")
            lh_color = (0, 255, 0) if results.left_hand_landmarks else (0, 0, 255)
            rh_color = (0, 255, 0) if results.right_hand_landmarks else (0, 0, 255)
            cv2.putText(image, lh_status, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lh_color, 2)
            cv2.putText(image, rh_status, (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rh_color, 2)
            
            cv2.putText(image, "Press SPACE when ready, Q to skip", (10, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow('Data Collection', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                return None
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Starting in {i}...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
        
        # Capture sequence
        hand_detected_count = 0
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            keypoints = self.extract_keypoints(results)
            frames.append(keypoints)
            
            # Track hand detection
            if results.right_hand_landmarks or results.left_hand_landmarks:
                hand_detected_count += 1
            
            # Draw landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
            # Show progress
            cv2.rectangle(image, (0, 0), (640, 60), (0, 255, 0), -1)
            cv2.putText(image, f"RECORDING: {frame_idx+1}/{SEQUENCE_LENGTH}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            cv2.imshow('Data Collection', image)
            cv2.waitKey(1)
        
        # Check quality
        is_good, message = self.check_sequence_quality(frames, sign)
        
        print(f"      {message}")
        
        if is_good:
            return np.array(frames)
        else:
            print(f"      ⚠️  Sequence rejected. Please try again with better hand visibility.")
            return None
    
    def save_sequence(self, sign, sequence, seq_num):
        """Save a sequence to disk."""
        sign_path = DATA_PATH / sign
        
        # Find next available sequence number
        existing_nums = [int(f.name) for f in sign_path.iterdir() if f.is_dir() and f.name.isdigit()]
        new_seq_num = max(existing_nums) + 1 if existing_nums else 1
        
        seq_path = sign_path / str(new_seq_num)
        seq_path.mkdir(parents=True, exist_ok=True)
        
        for frame_num, frame in enumerate(sequence, 1):
            np.save(str(seq_path / f"{frame_num}.npy"), frame)
        
        print(f"      ✓ Saved as sequence #{new_seq_num}")
        return new_seq_num
    
    def collect_for_sign(self, sign, cap):
        """Collect all sequences for a single sign."""
        print(f"\n{'='*60}")
        print(f"COLLECTING DATA FOR: {sign}")
        print(f"{'='*60}")
        
        # Show existing data info
        self.show_training_data_example(sign, cap)
        
        info = PROBLEM_SIGNS_TO_COLLECT[sign]
        print(f"\n   📌 This sign is often confused with: {', '.join(info['confused_with'])}")
        print(f"   💡 Tip: {info['tip']}")
        
        collected = 0
        attempts = 0
        max_attempts = SEQUENCES_PER_SIGN * 3  # Allow some retries
        
        while collected < SEQUENCES_PER_SIGN and attempts < max_attempts:
            attempts += 1
            sequence = self.collect_sequence(sign, cap, collected + 1, SEQUENCES_PER_SIGN)
            
            if sequence is not None:
                self.save_sequence(sign, sequence, collected + 1)
                collected += 1
        
        print(f"\n   ✓ Collected {collected}/{SEQUENCES_PER_SIGN} sequences for '{sign}'")
        return collected
    
    def run_collection(self):
        """Run the full data collection process."""
        print("\n" + "="*60)
        print("TARGETED DATA COLLECTION FOR PROBLEM SIGNS")
        print("="*60)
        print("\nThis will collect additional training data for signs that")
        print("performed poorly in the diagnostic test.")
        print(f"\nSigns to collect: {list(PROBLEM_SIGNS_TO_COLLECT.keys())}")
        print(f"Sequences per sign: {SEQUENCES_PER_SIGN}")
        
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            # Show instructions
            print("\nPress SPACE to start, Q to quit...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                cv2.rectangle(frame, (0, 0), (640, 150), (0, 0, 0), -1)
                cv2.putText(frame, "TARGETED DATA COLLECTION", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Will collect {SEQUENCES_PER_SIGN} sequences per sign", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, "Press SPACE to start", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    return
            
            # Collect data for each problem sign
            results = {}
            for sign in PROBLEM_SIGNS_TO_COLLECT.keys():
                # Ask if user wants to collect for this sign
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
                    cv2.putText(frame, f"Next sign: {sign}", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, "SPACE=Collect | S=Skip | Q=Quit", (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Data Collection', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        collected = self.collect_for_sign(sign, cap)
                        results[sign] = collected
                        break
                    elif key == ord('s'):
                        print(f"\n   Skipping '{sign}'...")
                        results[sign] = 0
                        break
                    elif key == ord('q'):
                        print("\nCollection ended by user.")
                        self._print_summary(results)
                        return
            
            self._print_summary(results)
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _print_summary(self, results):
        """Print collection summary."""
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        
        total = 0
        for sign, count in results.items():
            status = "✓" if count >= SEQUENCES_PER_SIGN else ("~" if count > 0 else "✗")
            print(f"   {status} {sign}: {count} sequences")
            total += count
        
        print(f"\n   Total new sequences: {total}")
        print("\n   Next steps:")
        print("   1. Run 6_optimize_model.py to retrain with new data")
        print("   2. Run 5_diagnostic_test.py to verify improvement")


def main():
    collector = QualityDataCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()
