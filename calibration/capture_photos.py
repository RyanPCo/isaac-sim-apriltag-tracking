import cv2
import os
import time
import argparse

def capture_calibration_images(camera_index=0):
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        return
    
    # Try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Get and print camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Create calibration directory if it doesn't exist
    calib_dir = "./hand"
    os.makedirs(calib_dir, exist_ok=True)
    
    # Create resizable window
    cv2.namedWindow('Webcam - Press Enter to Capture', cv2.WINDOW_NORMAL)
    
    print(f"Webcam stream started (camera index: {camera_index}). Press 'Enter' to capture a photo, 'q' to quit")
    print(f"Photos will be saved to: {calib_dir}")
    
    photo_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam - Press Enter to Capture', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 13:  # Enter key
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"calib_{timestamp}_{photo_count:02d}.jpg"
            filepath = os.path.join(calib_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            print(f"Captured: {filename}")
            photo_count += 1
            
            # Show confirmation
            cv2.putText(frame, "CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam - Press Enter to Capture', frame)
            cv2.waitKey(1000)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {photo_count} calibration images")
    print(f"Images saved to: {calib_dir}")
    print("You can now run camera_calibration.py with these images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture calibration images from webcam')
    parser.add_argument('--camera', '-c', type=int, default=1, 
                       help='Camera index (default: 1)')
    
    args = parser.parse_args()
    capture_calibration_images(args.camera)
