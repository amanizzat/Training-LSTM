"""
Master Setup and Training Script for Sign Language Recognition
This script guides you through the entire process step by step.

Scripts:
  0_setup_and_run.py    - This file (setup guide)
  1_collect_videos.py   - Record sign language videos
  2_extract_landmarks.py - Extract pose/hand landmarks
  3_train_model.py      - Train LSTM model (mobile-compatible)
  4_evaluate_model.py   - Check accuracy and bias
  5_test_live.py        - Test with webcam
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("\n⚠ Warning: Python 3.8 or higher is recommended")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_requirements():
    """Install required Python packages."""
    print_header("INSTALLING REQUIRED PACKAGES")
    
    if not Path('requirements.txt').exists():
        print("Error: requirements.txt not found!")
        print("Make sure all files are in the same directory.")
        return False
    
    print("Installing packages... This may take a few minutes.\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Error installing packages")
        return False

def run_script(script_name, description):
    """Run a Python script."""
    print_header(description)
    
    if not Path(script_name).exists():
        print(f"Error: {script_name} not found!")
        return False
    
    print(f"Running {script_name}...\n")
    
    try:
        subprocess.check_call([sys.executable, script_name])
        print(f"\n✓ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print(f"\n✗ Error running {script_name}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠ {script_name} interrupted by user")
        return False

def main():
    """Main execution flow."""
    print_header("SIGN LANGUAGE MODEL TRAINING SETUP")
    
    print("This script will guide you through:")
    print("  1. Installing required packages")
    print("  2. Collecting sign language videos")
    print("  3. Extracting landmarks from videos")
    print("  4. Training the AI model (mobile-compatible)")
    print("  5. Evaluating model accuracy")
    print("  6. Testing with webcam")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 0: Check Python version
    if not check_python_version():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Step 1: Install requirements
    print_header("STEP 1: PACKAGE INSTALLATION")
    print("Installing required Python packages...")
    response = input("Proceed with installation? (y/n): ")
    
    if response.lower() == 'y':
        if not install_requirements():
            print("\nSetup failed. Please install packages manually:")
            print("  pip install -r requirements.txt")
            return
    else:
        print("Skipping package installation.")
        print("Make sure you have all required packages installed!")
    
    # Step 2: Collect videos
    print_header("STEP 2: VIDEO COLLECTION")
    print("This will open your webcam to collect sign language videos.")
    print("\nBefore starting:")
    print("  - Make sure you have good lighting")
    print("  - Use a simple background")
    print("  - Be prepared to perform each sign 30 times")
    print("  - Each recording will be 2 seconds long")
    
    response = input("\nStart video collection? (y/n): ")
    
    if response.lower() == 'y':
        if not run_script('1_collect_videos.py', 'COLLECTING SIGN VIDEOS'):
            print("\nVideo collection incomplete.")
            response = input("Continue to next step anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("Skipping video collection.")
        print("Make sure you have videos in: C:/fyp/SL_Data/")
    
    # Step 3: Extract landmarks
    print_header("STEP 3: LANDMARK EXTRACTION")
    print("This will process your videos and extract pose landmarks.")
    
    response = input("\nStart landmark extraction? (y/n): ")
    
    if response.lower() == 'y':
        if not run_script('2_extract_landmarks.py', 'EXTRACTING LANDMARKS'):
            print("\nLandmark extraction incomplete.")
            response = input("Continue to next step anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("Skipping landmark extraction.")
        print("Make sure you have processed data in: C:/fyp/SL_Data_Processed/")
    
    # Step 4: Train model
    print_header("STEP 4: MODEL TRAINING")
    print("This will train a mobile-compatible LSTM model.")
    print("Training may take 10-30 minutes depending on your hardware.")
    print("\nModel features:")
    print("  - Simple LSTM (works on mobile TFLite)")
    print("  - Data augmentation for better accuracy")
    print("  - Automatic TFLite conversion")
    
    response = input("\nStart model training? (y/n): ")
    
    if response.lower() == 'y':
        if not run_script('3_train_model.py', 'TRAINING MODEL'):
            print("\nModel training incomplete.")
            response = input("Continue to next step anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("Skipping model training.")
        print("Make sure you have action_model.keras and action_model.tflite!")
    
    # Step 5: Evaluate model
    print_header("STEP 5: MODEL EVALUATION")
    print("This will check your model's accuracy for each sign.")
    
    response = input("\nEvaluate model? (y/n): ")
    
    if response.lower() == 'y':
        if not run_script('4_evaluate_model.py', 'EVALUATING MODEL'):
            print("\nEvaluation incomplete.")
    else:
        print("Skipping evaluation.")
    
    # Step 6: Test live
    print_header("STEP 6: LIVE TESTING")
    print("This will test your model with webcam.")
    
    response = input("\nTest with webcam? (y/n): ")
    
    if response.lower() == 'y':
        if not run_script('5_test_live.py', 'TESTING MODEL LIVE'):
            print("\nLive testing ended.")
    else:
        print("Skipping live test.")
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    print("✓ All steps completed!")
    print("\n" + "="*70)
    print("FLUTTER DEPLOYMENT")
    print("="*70)
    print("\n1. Copy files to your Flutter project:")
    print("   - action_model.tflite → C:/fyp/lang_bridge/assets/")
    print("   - action_labels.txt   → C:/fyp/lang_bridge/assets/")
    
    print("\n2. Update pubspec.yaml:")
    print("   flutter:")
    print("     assets:")
    print("       - assets/action_model.tflite")
    print("       - assets/action_labels.txt")
    
    print("\n3. Run your Flutter app:")
    print("   cd C:/fyp/lang_bridge")
    print("   flutter clean")
    print("   flutter pub get")
    print("   flutter run")
    
    print("\n" + "="*70)
    print("\nGood luck with your project! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()