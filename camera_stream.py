import cv2
import numpy as np
import threading
import time
from apriltag_tracker import AprilTagTracker
from hand_pose import HandPoseDetector

class CameraStream:
    def __init__(self, cam_index=1, width=1920, height=1080, tag_size=0.07, calib_path="./calibration/webcam/calib.npz"):
        self.cap = cv2.VideoCapture(cam_index)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Default intrinsics (will be overridden by calibration)
        K = np.array([[width,0,width//2],[0,height,height//2],[0,0,1]], dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)

        # Try to load calibration if available
        try:
            if calib_path is not None:
                data = np.load(calib_path)
                if "mtx" in data:
                    K = data["mtx"].astype(np.float32)
                if "dist" in data:
                    dist_raw = data["dist"]
                    dist = dist_raw.astype(np.float32).reshape(-1)
        except Exception as e:
            print(f"[WARNING]: Could not load calibration from {calib_path}: {e}")
        
        self.K = K
        self.dist = dist
        self._undistort_enabled = np.any(np.abs(self.dist) > 1e-12)
        self._undistort_initialized = False
        self._map1 = None
        self._map2 = None
        self._K_rectified = self.K.copy()
        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_undistorted_frame = None
        self.running = False
        self.thread = None
        
        # Initialize AprilTag tracker
        self.apriltag_tracker = AprilTagTracker(tag_size=tag_size)
        
        # Initialize hand pose detector
        self.hand_pose_detector = HandPoseDetector()
        
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok: 
                time.sleep(0.01)
                continue
                
            # Lazy initialize undistortion maps when we know frame size
            if self._undistort_enabled and not self._undistort_initialized:
                h, w = frame.shape[:2]
                newK, _roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
                self._K_rectified = newK.astype(np.float32)
                self._map1, self._map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
                self._undistort_initialized = True

            # Store latest frame
            with self.lock:
                self.latest_frame = frame.copy()
                
            # Create undistorted frame for AprilTag detection
            if self._undistort_enabled and self._undistort_initialized:
                undistorted_frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.latest_undistorted_frame = undistorted_frame.copy()
            else:
                with self.lock:
                    self.latest_undistorted_frame = frame.copy()
                    
            # Detect hands and AprilTags in the undistorted frame with camera parameters for 3D pose
            if self.latest_undistorted_frame is not None:
                camera_params = self.get_camera_params()
                self.hand_pose_detector.detect_hands(self.latest_undistorted_frame, camera_params)
                self.apriltag_tracker.detect_tags(self.latest_undistorted_frame, camera_params)
                    
            # Small sleep to reduce CPU
            time.sleep(0.005)
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
    
    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_latest_undistorted_frame(self):
        with self.lock:
            return self.latest_undistorted_frame.copy() if self.latest_undistorted_frame is not None else None
    
    def get_camera_params(self):
        """Get camera parameters for AprilTag detection"""
        if self._undistort_enabled and self._undistort_initialized:
            return self._K_rectified
        else:
            return self.K
    
    def get_undistortion_maps(self):
        """Get undistortion maps if available"""
        if self._undistort_enabled and self._undistort_initialized:
            return self._map1, self._map2
        else:
            return None, None
    
    def show_video(self, window_name="Camera Stream"):
        """Display the latest frame with AprilTag detections"""
        try:
            # Make window resizable
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)  # Set initial size to 1280x720
            
            # Get latest frame and use cached detection results
            frame = self.get_latest_undistorted_frame()
            if frame is not None:
                # Get cached AprilTag detection results instead of re-detecting
                apriltag_data = self.apriltag_tracker.get_latest()
                
                if apriltag_data is not None:
                    # Draw detections on frame
                    frame = self.apriltag_tracker.draw_detections(
                        frame, 
                        apriltag_data["detections"],
                        apriltag_data.get("R_cam_tag"),
                        apriltag_data.get("t_cam_tag"),
                        self.get_camera_params()
                    )
                
                # Draw hand pose detections
                hand_detection = self.hand_pose_detector.get_latest()
                if hand_detection is not None:
                    frame = self.hand_pose_detector.draw_detections(frame, hand_detection)
                
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
            return True
        except Exception as e:
            print(f"[WARNING]: Error displaying camera video: {e}")
            return True


if __name__ == "__main__":
    # Test the camera stream and AprilTag detection
    print("[INFO]: Starting CameraStream test...")
    print("[INFO]: Press 'q' in the video window to stop")
    
    # Initialize camera stream
    camera_stream = CameraStream(
        cam_index=1,  # Change to 0 if using built-in webcam
        width=1920, 
        height=1080, 
        tag_size=0.07,  # 7cm tag size
        calib_path="./calibration/webcam/calib.npz"
    )
    
    try:
        # Start the camera stream
        camera_stream.start()
        print("[INFO]: Camera stream started successfully")
        print("[INFO]: Coordinates displayed on video - press 'q' to quit")
        
        # Main loop to display video (coordinates shown on video, minimal console output)
        while True:
            # Display video with AprilTag and hand detections
            if not camera_stream.show_video("Camera Stream Test - AprilTag + Hand Pose"):
                print("[INFO]: Video window closed. Stopping...")
                break
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("[INFO]: Interrupted by user")
    except Exception as e:
        print(f"[ERROR]: Unexpected error: {e}")
    finally:
        # Clean up
        camera_stream.stop()
        if hasattr(camera_stream, 'hand_pose_detector'):
            camera_stream.hand_pose_detector.cleanup()
        cv2.destroyAllWindows()
        print("[INFO]: Camera stream stopped and cleaned up")
