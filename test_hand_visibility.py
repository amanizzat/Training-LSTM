import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np

# --- Setup ---
VIDEO_PATH = Path("C:/fyp/SL_Data")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def test_video_detection(video_path):
    """Test what MediaPipe detects in a video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Stats
    pose_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0
    
    print(f"\nTesting: {video_path.name}")
    print(f"Total frames: {frame_count}")
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,  # Lower threshold
        min_tracking_confidence=0.3,
        model_complexity=1
    ) as holistic:
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Count detections
            if results.pose_landmarks:
                pose_detected += 1
            if results.left_hand_landmarks:
                left_hand_detected += 1
            if results.right_hand_landmarks:
                right_hand_detected += 1
            
            # Visualize every 10th frame
            if frame_num % 10 == 0:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw pose
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # Draw hands
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Show status
                cv2.putText(image_bgr, f"Frame: {frame_num}/{frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_bgr, f"Pose: {'YES' if results.pose_landmarks else 'NO'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_bgr, f"Left Hand: {'YES' if results.left_hand_landmarks else 'NO'}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_bgr, f"Right Hand: {'YES' if results.right_hand_landmarks else 'NO'}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Hand Detection Test', image_bgr)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
            frame_num += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Pose:       {pose_detected}/{frame_count} frames ({pose_detected/frame_count*100:.1f}%)")
    print(f"  Left Hand:  {left_hand_detected}/{frame_count} frames ({left_hand_detected/frame_count*100:.1f}%)")
    print(f"  Right Hand: {right_hand_detected}/{frame_count} frames ({right_hand_detected/frame_count*100:.1f}%)")
    
    # Recommendations
    print("\n" + "="*60)
    if left_hand_detected < frame_count * 0.5 or right_hand_detected < frame_count * 0.5:
        print("⚠ WARNING: Hands not detected in >50% of frames!")
        print("\nRecommendations:")
        print("  1. ✋ Keep hands IN FRAME at all times")
        print("  2. 💡 Ensure good lighting (avoid shadows)")
        print("  3. 🎨 Use plain background (avoid patterns)")
        print("  4. 📏 Keep hands 30-100cm from camera")
        print("  5. 🖐️ Show fingers clearly (spread out)")
        print("  6. ⚡ Move hands slower during recording")
    else:
        print("✓ Good hand detection! Hands visible in most frames.")
    print("="*60)

def test_all_actions():
    """Test first video of each action."""
    print("\n" + "="*70)
    print("HAND DETECTION TEST")
    print("="*70)
    
    try:
        actions = sorted([f.name for f in VIDEO_PATH.iterdir() if f.is_dir()])
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"\nFound {len(actions)} actions: {actions}")
    print("Testing first video of each action...")
    print("Press 'q' to skip to next video\n")
    
    for action in actions:
        video_folder = VIDEO_PATH / action
        video_files = sorted(list(video_folder.glob('*.mp4')))
        
        if video_files:
            test_video_detection(video_files[0])
            
            input("\nPress Enter to test next action...")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    try:
        test_all_actions()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()