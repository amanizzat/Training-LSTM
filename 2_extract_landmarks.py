"""
STEP 2: EXTRACT LANDMARKS FROM VIDEOS
======================================
Extracts 258 features (Pose + Hands) from video files.

Structure:
- Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132 features
- Left Hand: 21 landmarks × 3 values (x, y, z) = 63 features  
- Right Hand: 21 landmarks × 3 values (x, y, z) = 63 features
- Total: 258 features per frame

Usage:
    python 2_extract_landmarks.py
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
VIDEO_PATH = Path("C:/fyp/SL_Data")          # Input: folder with video subfolders
DATA_PATH = Path("C:/fyp/SL_Data_Processed") # Output: processed landmarks

SEQUENCE_LENGTH = 30  # Frames per video
NUM_FEATURES = 258    # Pose(132) + LeftHand(63) + RightHand(63)
# ---------------------

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Extract 258 features from MediaPipe results.
    Order: [Pose(132), LeftHand(63), RightHand(63)]
    """
    # Pose: 33 landmarks × 4 values = 132
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                        for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    
    # Left Hand: 21 landmarks × 3 values = 63
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] 
                             for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # Right Hand: 21 landmarks × 3 values = 63
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] 
                              for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    return np.concatenate([pose, left_hand, right_hand])


def process_video(video_path, output_folder, holistic):
    """Process a single video and save landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return False
    
    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    landmarks_sequence = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            landmarks_sequence.append(np.zeros(NUM_FEATURES))
            continue
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        landmarks_sequence.append(keypoints)
    
    cap.release()
    
    # Pad if needed
    while len(landmarks_sequence) < SEQUENCE_LENGTH:
        landmarks_sequence.append(np.zeros(NUM_FEATURES))
    
    # Save each frame
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, keypoints in enumerate(landmarks_sequence[:SEQUENCE_LENGTH]):
        np.save(str(output_folder / f"{i + 1}.npy"), keypoints)
    
    return True


def get_next_sequence_number(output_folder):
    """
    Get the next available sequence number for a sign.
    Checks existing folders in SL_Data_Processed and returns max + 1.
    """
    if not output_folder.exists():
        return 1
    
    existing_sequences = [
        int(f.name) for f in output_folder.iterdir() 
        if f.is_dir() and f.name.isdigit()
    ]
    
    return max(existing_sequences) + 1 if existing_sequences else 1


def main():
    print("\n" + "=" * 60)
    print("STEP 2: LANDMARK EXTRACTION (Auto-Increment Mode)")
    print("=" * 60)
    print(f"Input:  {VIDEO_PATH}")
    print(f"Output: {DATA_PATH}")
    print(f"Features: {NUM_FEATURES} per frame")
    print(f"Frames: {SEQUENCE_LENGTH} per video")
    print("=" * 60)
    
    # Find all action folders
    if not VIDEO_PATH.exists():
        print(f"\n✗ Video path not found: {VIDEO_PATH}")
        return
    
    actions = sorted([f.name for f in VIDEO_PATH.iterdir() if f.is_dir()])
    
    if not actions:
        print("\n✗ No action folders found!")
        return
    
    print(f"\n✓ Found {len(actions)} actions: {actions}")
    
    # Process each action
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        total_videos = 0
        successful = 0
        
        for action in actions:
            video_folder = VIDEO_PATH / action
            output_folder = DATA_PATH / action
            
            # Get starting sequence number (auto-increment from existing)
            next_seq_num = get_next_sequence_number(output_folder)
            print(f"\n📁 {action}: Starting from sequence {next_seq_num}")
            
            # Find videos
            videos = sorted(
                list(video_folder.glob("*.mp4")) + list(video_folder.glob("*.avi")),
                key=lambda f: int(f.stem) if f.stem.isdigit() else 0
            )
            
            print(f"   Found {len(videos)} new videos to process")
            
            for video_file in tqdm(videos, desc=f"   Processing"):
                total_videos += 1
                try:
                    # Use auto-incremented sequence number instead of filename
                    seq_folder = output_folder / str(next_seq_num)
                    
                    if process_video(video_file, seq_folder, holistic):
                        successful += 1
                        next_seq_num += 1  # Increment for next video
                except Exception as e:
                    print(f"\n   ✗ Error on {video_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"✓ Processed: {successful}/{total_videos} videos")
    print(f"✓ Output: {DATA_PATH}")
    print("\nNext: Run python 3_train_model.py")


if __name__ == "__main__":
    main()