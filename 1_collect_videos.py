import cv2
import os
from pathlib import Path
import time
import mediapipe as mp

# --- Configuration ---
VIDEO_PATH = Path("C:/fyp/SL_Data")

# Actions to collect (leave empty to auto-detect from folders)
actions = []  # Empty = auto-detect, or specify like: ["A", "B", "C"]

# TARGET: Total number of videos you want per action
target_dataset_size = 50

# Video recording settings
fps = 30
duration = 2  # seconds per video

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

def draw_styled_landmarks(image, results):
    """Draws full skeleton (pose, hands, face) with custom styling."""
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def get_action_status():
    """Get current status of each action folder."""
    status = {}
    
    for action in actions:
        action_path = VIDEO_PATH / action
        
        # Create folder if it doesn't exist
        if not action_path.exists():
            action_path.mkdir(parents=True, exist_ok=True)
            existing_count = 0
        else:
            # Count existing videos
            existing_videos = list(action_path.glob('*.mp4'))
            existing_count = len(existing_videos)
        
        needed = max(0, target_dataset_size - existing_count)
        status[action] = {
            'existing': existing_count,
            'needed': needed,
            'target': target_dataset_size,
            'complete': existing_count >= target_dataset_size
        }
    
    return status

def record_videos():
    global actions
    
    # Open camera
    cap = cv2.VideoCapture(1)  # Change to 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # --- Handle Manual vs Auto Detection ---
    if len(actions) > 0:
        print(f"Using manual action list: {actions}")
    else:
        print(f"Scanning for action folders in: {VIDEO_PATH}")
        try:
            actions = sorted([f.name for f in VIDEO_PATH.iterdir() if f.is_dir()])
            if not actions:
                print(f"ERROR: No action folders found in {VIDEO_PATH}")
                print("Please create folders for your signs first (e.g., A, B, C)")
                return
            print(f"✓ Auto-detected {len(actions)} actions: {actions}")
        except Exception as e:
            print(f"Error scanning folders: {e}")
            return
    
    # Get initial status
    status = get_action_status()
    
    # Show summary
    print("\n" + "="*70)
    print("SIGN LANGUAGE DATA COLLECTION - TARGET BASED")
    print("="*70)
    print(f"Target per action: {target_dataset_size} videos")
    print("\nCurrent Status:")
    print("-" * 70)
    
    total_existing = 0
    total_needed = 0
    
    for action in actions:
        info = status[action]
        status_icon = "✓" if info['complete'] else "○"
        print(f"  {status_icon} {action:15} | Existing: {info['existing']:3} | Need: {info['needed']:3} | Target: {info['target']}")
        total_existing += info['existing']
        total_needed += info['needed']
    
    print("-" * 70)
    print(f"  TOTAL:          | Existing: {total_existing:3} | Need: {total_needed:3} | Target: {len(actions) * target_dataset_size}")
    print("="*70)
    
    if total_needed == 0:
        print("\n✓ All actions have reached the target! No videos needed.")
        response = input("\nDo you want to continue recording anyway? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            return
    
    print("\nInstructions:")
    print("  - Press SPACEBAR to start recording countdown")
    print("  - Press 'q' to quit at any time")
    print("  - Press 's' to skip current action")
    print()
    
    input("Press Enter to begin...")
    
    # Start collection
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions:
            info = status[action]
            
            if info['needed'] == 0:
                print(f"\n{'='*70}")
                print(f"✓ Skipping '{action}' - Already has {info['existing']}/{target_dataset_size} videos")
                print(f"{'='*70}")
                continue
            
            print(f"\n{'='*70}")
            print(f"Collecting videos for: '{action}'")
            print(f"Progress: {info['existing']}/{target_dataset_size} (need {info['needed']} more)")
            print(f"{'='*70}\n")
            
            action_path = VIDEO_PATH / action
            
            # Start from where we left off
            start_num = info['existing'] + 1
            end_num = target_dataset_size
            
            for video_num in range(start_num, end_num + 1):
                print(f"\nVideo {video_num}/{target_dataset_size} for '{action}'")
                print("Press SPACEBAR to start | 's' to skip action | 'q' to quit")
                
                # Wait for user input
                waiting = True
                skip_action = False
                
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # Display info
                    cv2.putText(image, f"Action: {action}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Video: {video_num}/{target_dataset_size}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Remaining: {target_dataset_size - video_num + 1}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(image, "SPACE=Start | S=Skip | Q=Quit", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Progress bar
                    progress_width = int((video_num - 1) / target_dataset_size * 600)
                    cv2.rectangle(image, (20, 450), (620, 470), (50, 50, 50), -1)  # Background
                    cv2.rectangle(image, (20, 450), (20 + progress_width, 470), (0, 255, 0), -1)  # Progress
                    
                    # Progress text
                    progress_text = f"{video_num - 1}/{target_dataset_size}"
                    cv2.putText(image, progress_text, (640 - 120, 440), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Data Collection', image)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        waiting = False
                    elif key == ord('s'):
                        waiting = False
                        skip_action = True
                        print(f"Skipping remaining videos for '{action}'")
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\nCollection stopped by user.")
                        return
                
                if skip_action:
                    break
                
                # Countdown
                print("Get ready...")
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    text = str(countdown)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.imshow('Data Collection', image)
                    cv2.waitKey(1000)
                
                # Show GO
                ret, frame = cap.read()
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, "GO!", (frame.shape[1]//2 - 70, frame.shape[0]//2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                    cv2.imshow('Data Collection', image)
                    cv2.waitKey(500)
                
                # Start recording
                video_file_path = action_path / f"{video_num}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_file_path), fourcc, fps, (640, 480))
                
                print(f"Recording... Perform sign '{action}' NOW!")
                
                frames_to_record = int(fps * duration)
                for frame_num in range(frames_to_record):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process for display (with landmarks)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # Recording indicator
                    cv2.circle(image, (620, 20), 10, (0, 0, 255), -1)
                    cv2.putText(image, "REC", (560, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(image, f"Action: {action}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Recording progress bar
                    rec_progress = int(((frame_num + 1) / frames_to_record) * 600)
                    cv2.rectangle(image, (20, 450), (620, 470), (50, 50, 50), -1)
                    cv2.rectangle(image, (20, 450), (20 + rec_progress, 470), (0, 255, 0), -1)
                    
                    # IMPORTANT: Save CLEAN frame without landmarks!
                    # Landmarks will be re-extracted in step 2
                    out.write(frame)
                    
                    # Show preview with landmarks (for user feedback)
                    cv2.imshow('Data Collection', image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        out.release()
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\nCollection stopped by user.")
                        return
                
                out.release()
                print(f"✓ Saved: {video_file_path}")
                
                # Brief pause
                time.sleep(0.5)
            
            print(f"\n✓ Completed all videos for '{action}' ({target_dataset_size}/{target_dataset_size})")
            time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    final_status = get_action_status()
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print("="*70)
    print("\nFinal Status:")
    print("-" * 70)
    
    all_complete = True
    for action in actions:
        info = final_status[action]
        status_icon = "✓" if info['complete'] else "○"
        print(f"  {status_icon} {action:15} | Videos: {info['existing']:3}/{info['target']}")
        if not info['complete']:
            all_complete = False
    
    print("-" * 70)
    
    if all_complete:
        print("\n🎉 SUCCESS! All actions have reached the target of {target_dataset_size} videos!")
    else:
        print(f"\n⚠ Some actions still need more videos to reach {target_dataset_size}")
    
    print("="*70)

# --- Main ---
if __name__ == "__main__":
    try:
        record_videos()
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()