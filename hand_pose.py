import cv2
import numpy as np
import mediapipe as mp
import json
from typing import Optional, Tuple, Dict, Any

class HandPoseDetector:
    def __init__(self, model_path: str = "./models/hand_landmarker.task", 
                 measurements_path: str = "./calibration/hand/measurements.json"):
        """
        Initialize hand pose detector with MediaPipe model and hand measurements
        
        Args:
            model_path: Path to the MediaPipe hand landmarker task file
            measurements_path: Path to the hand measurements JSON file
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand landmarker
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load hand measurements and template
        self.measurements = self._load_measurements(measurements_path)
        
        # Hand landmark indices for key points (only those in measurements)
        self.landmark_indices = self.measurements['landmark_ids']
        
        # Palm template in meters (3D world coordinates)
        self.palm_template = np.array([
            self.measurements['palm_template_m']['wrist'],
            self.measurements['palm_template_m']['thumb_cmc'],
            self.measurements['palm_template_m']['index_mcp'],
            self.measurements['palm_template_m']['middle_mcp'],
            self.measurements['palm_template_m']['ring_mcp'],
            self.measurements['palm_template_m']['pinky_mcp']
        ], dtype=np.float32)
        
        self.latest_detection = None
        
    def _load_measurements(self, measurements_path: str) -> Dict[str, Any]:
        """Load hand measurements from JSON file"""
        try:
            with open(measurements_path, 'r') as f:
                measurements = json.load(f)
            print(f"[INFO]: Loaded hand measurements from {measurements_path}")
            return measurements
        except Exception as e:
            print(f"[WARNING]: Could not load measurements from {measurements_path}: {e}")
            # Fallback to default measurements
            return {
                "landmark_ids": {
                    "wrist": 0,
                    "thumb_cmc": 1,
                    "index_mcp": 5,
                    "middle_mcp": 9,
                    "ring_mcp": 13,
                    "pinky_mcp": 17
                },
                "palm_template_m": {
                    "wrist": [0.0, 0.0, 0.0],
                    "thumb_cmc": [-0.04, -0.02, 0.0],
                    "index_mcp": [-0.036, -0.09, 0.0],
                    "middle_mcp": [-0.013, -0.093, 0.0],
                    "ring_mcp": [0.007, -0.087, 0.0],
                    "pinky_mcp": [0.025, -0.077, 0.0]
                }
            }
        
    def detect_hands(self, image: np.ndarray, camera_matrix: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Detect hands in the given image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detection results or None if no hands detected
        """
        if image is None:
            return None
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
            
        detections = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 2D landmarks
            landmarks_2d = []
            for landmark in hand_landmarks.landmark:
                h, w, _ = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks_2d.append([x, y])
            
            # Extract 3D landmarks (normalized coordinates)
            landmarks_3d = []
            for landmark in hand_landmarks.landmark:
                landmarks_3d.append([landmark.x, landmark.y, landmark.z])
            
            # Get hand handedness
            handedness = None
            if results.multi_handedness:
                for handedness_info in results.multi_handedness:
                    if handedness_info.classification:
                        handedness = handedness_info.classification[0].label
            
            # Solve for absolute 3D world coordinates using solvePnP
            world_coords_3d = None
            if camera_matrix is not None:
                world_coords_3d = self._solve_hand_pose(landmarks_2d, camera_matrix)
            
            # Calculate key measurements
            key_points = self._extract_key_points(landmarks_2d, landmarks_3d, world_coords_3d)
            
            detection = {
                'landmarks_2d': np.array(landmarks_2d),
                'landmarks_3d': np.array(landmarks_3d),
                'world_coords_3d': world_coords_3d,
                'handedness': handedness,
                'key_points': key_points,
                'bounding_box': self._calculate_bounding_box(landmarks_2d)
            }
            
            detections.append(detection)
        
        result = {
            'detections': detections,
            'num_hands': len(detections),
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        self.latest_detection = result
        return result
        
    def _solve_hand_pose(self, landmarks_2d: list, camera_matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Solve for hand pose using solvePnP with measured palm template
        
        Args:
            landmarks_2d: 2D image coordinates of detected landmarks
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Dictionary with rotation, translation, and world coordinates
        """
        try:
            # Extract only the landmarks that are in our measurements
            measured_landmarks_2d = []
            for landmark_name, landmark_id in self.landmark_indices.items():
                if landmark_id < len(landmarks_2d):
                    measured_landmarks_2d.append(landmarks_2d[landmark_id])
                else:
                    return None
            
            if len(measured_landmarks_2d) < 4:  # Need at least 4 points for solvePnP
                return None
            
            # Convert to numpy arrays
            image_points = np.array(measured_landmarks_2d, dtype=np.float32)
            object_points = self.palm_template
            
            # Solve PnP to find rotation and translation
            success, rotation_vec, translation_vec = cv2.solvePnP(
                object_points, image_points, camera_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            
            # Transform palm template to world coordinates
            world_coords = []
            for point_3d in object_points:
                # Apply rotation and translation
                point_world = rotation_matrix @ point_3d + translation_vec.flatten()
                world_coords.append(point_world)
            
            return {
                'rotation_matrix': rotation_matrix,
                'rotation_vector': rotation_vec.flatten(),
                'translation': translation_vec.flatten(),
                'world_coords': np.array(world_coords),
                'success': True
            }
            
        except Exception as e:
            print(f"[WARNING]: Error solving hand pose: {e}")
            return None
        
    def _extract_key_points(self, landmarks_2d: list, landmarks_3d: list, world_coords_3d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract key hand measurements and points"""
        key_points = {}
        
        # Extract specific landmark points (only those in measurements)
        for name, idx in self.landmark_indices.items():
            if idx < len(landmarks_2d):
                key_points[f'{name}_2d'] = landmarks_2d[idx]
                key_points[f'{name}_3d'] = landmarks_3d[idx]
        
        # Add world coordinates if available
        if world_coords_3d is not None and world_coords_3d.get('success', False):
            world_coords = world_coords_3d['world_coords']
            for i, (name, idx) in enumerate(self.landmark_indices.items()):
                if i < len(world_coords):
                    key_points[f'{name}_world'] = world_coords[i]
            
            # Add pose information
            key_points['rotation_matrix'] = world_coords_3d['rotation_matrix']
            key_points['translation'] = world_coords_3d['translation']
            
            # Calculate palm measurements in world coordinates
            if len(world_coords) >= 6:
                # Palm width (distance between index and pinky MCP)
                index_mcp_world = world_coords[2]  # index_mcp
                pinky_mcp_world = world_coords[5]  # pinky_mcp
                palm_width_world = np.linalg.norm(index_mcp_world - pinky_mcp_world)
                key_points['palm_width_world'] = palm_width_world
                
                # Palm length (distance from wrist to middle MCP)
                wrist_world = world_coords[0]  # wrist
                middle_mcp_world = world_coords[3]  # middle_mcp
                palm_length_world = np.linalg.norm(middle_mcp_world - wrist_world)
                key_points['palm_length_world'] = palm_length_world
        
        return key_points
    
    def _calculate_bounding_box(self, landmarks_2d: list) -> Tuple[int, int, int, int]:
        """Calculate bounding box around hand landmarks"""
        if not landmarks_2d:
            return (0, 0, 0, 0)
            
        x_coords = [point[0] for point in landmarks_2d]
        y_coords = [point[1] for point in landmarks_2d]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_detections(self, image: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand detections on the image with coordinate display
        
        Args:
            image: Input image
            detection_result: Result from detect_hands()
            
        Returns:
            Image with detections drawn
        """
        if detection_result is None or 'detections' not in detection_result:
            return image
            
        output_image = image.copy()
        
        # Draw coordinate information in top-left corner
        output_image = self._draw_hand_coordinates(output_image, detection_result)
        
        for detection in detection_result['detections']:
            landmarks_2d = detection['landmarks_2d']
            handedness = detection['handedness']
            bounding_box = detection['bounding_box']
            
            # Draw landmarks (only measured ones)
            for name, idx in self.landmark_indices.items():
                if idx < len(landmarks_2d):
                    landmark = landmarks_2d[idx]
                    cv2.circle(output_image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
                    # Draw landmark name
                    cv2.putText(output_image, name, (landmark[0] + 5, landmark[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            # Draw connections
            self._draw_hand_connections(output_image, landmarks_2d)
            
            # Draw bounding box
            x, y, w, h = bounding_box
            color = (0, 255, 0) if handedness == 'Right' else (255, 0, 0)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw handedness label
            label = f"{handedness or 'Unknown'}"
            cv2.putText(output_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks_2d: list):
        """Draw connections between measured hand landmarks"""
        if len(landmarks_2d) < 6:
            return
            
        # Define connections between measured landmarks (palm structure)
        connections = [
            (0, 1),  # wrist to thumb_cmc
            (0, 2),  # wrist to index_mcp
            (0, 3),  # wrist to middle_mcp
            (0, 4),  # wrist to ring_mcp
            (0, 5),  # wrist to pinky_mcp
            (2, 3),  # index_mcp to middle_mcp
            (3, 4),  # middle_mcp to ring_mcp
            (4, 5),  # ring_mcp to pinky_mcp
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d):
                start_point = tuple(landmarks_2d[start_idx])
                end_point = tuple(landmarks_2d[end_idx])
                cv2.line(image, start_point, end_point, (255, 255, 255), 2)
    
    def _draw_hand_coordinates(self, image: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand coordinates on the image (similar to AprilTag coordinate display)
        Uses coordinates from solvePnP solution - shows only 3D world coordinates
        
        Args:
            image: Input image
            detection_result: Detection result from detect_hands()
            
        Returns:
            Image with coordinate information drawn
        """
        if detection_result is None or detection_result.get('num_hands', 0) == 0:
            return image
        
        # Start position for coordinate display (below AprilTag coordinates at y=30)
        start_y = 60  # Start below AprilTag coordinates
        current_y = start_y
        
        # Process each detected hand
        for i, detection in enumerate(detection_result['detections']):
            handedness = detection.get('handedness', 'Unknown')
            key_points = detection.get('key_points', {})
            world_coords_3d = detection.get('world_coords_3d')
            
            # Get 3D world coordinates from solvePnP solution
            if world_coords_3d is not None and world_coords_3d.get('success', False):
                # Get wrist world coordinates from solvePnP
                if 'wrist_world' in key_points:
                    wrist_world = key_points['wrist_world']
                    # Format similar to AprilTag: "Hand 1:[0.123 0.456 0.789] m"
                    world_text = f"Hand {i+1}:[{wrist_world[0]:.3f} {wrist_world[1]:.3f} {wrist_world[2]:.3f}] m"
                    cv2.putText(image, world_text, (10, current_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    current_y += 30
            else:
                # Show solvePnP failure status
                status_text = f"Hand {i+1}: solvePnP FAILED"
                cv2.putText(image, status_text, (10, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                current_y += 30
        
        return image
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the latest detection result"""
        return self.latest_detection
    
    def get_hand_pose_3d(self, detection_result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Extract 3D hand pose information with world coordinates
        
        Args:
            detection_result: Detection result (uses latest if None)
            
        Returns:
            Dictionary with 3D pose information including world coordinates
        """
        if detection_result is None:
            detection_result = self.latest_detection
            
        if detection_result is None or 'detections' not in detection_result:
            return None
            
        poses = []
        
        for detection in detection_result['detections']:
            handedness = detection['handedness']
            world_coords_3d = detection.get('world_coords_3d')
            
            if world_coords_3d is not None and world_coords_3d.get('success', False):
                # Use world coordinates for pose calculation
                world_coords = world_coords_3d['world_coords']
                rotation_matrix = world_coords_3d['rotation_matrix']
                translation = world_coords_3d['translation']
                
                # Calculate palm center in world coordinates
                palm_center = np.mean(world_coords, axis=0)
                
                # Calculate palm normal using world coordinates
                if len(world_coords) >= 6:
                    # Use measured landmarks for more accurate normal
                    index_mcp = world_coords[2]  # index_mcp
                    pinky_mcp = world_coords[5]  # pinky_mcp
                    middle_mcp = world_coords[3]  # middle_mcp
                    
                    palm_normal = np.cross(index_mcp - pinky_mcp, middle_mcp - palm_center)
                    if np.linalg.norm(palm_normal) > 1e-6:
                        palm_normal = palm_normal / np.linalg.norm(palm_normal)
                    else:
                        palm_normal = np.array([0, 0, 1])  # Default up direction
                    
                    pose_info = {
                        'handedness': handedness,
                        'palm_center': palm_center,
                        'palm_normal': palm_normal,
                        'rotation_matrix': rotation_matrix,
                        'translation': translation,
                        'world_coords': world_coords,
                        'success': True
                    }
                    
                    poses.append(pose_info)
        
        return {
            'poses': poses,
            'num_hands': len(poses),
            'timestamp': detection_result.get('timestamp', 0)
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


if __name__ == "__main__":
    # Test the hand pose detector
    print("[INFO]: Testing HandPoseDetector...")
    
    # Initialize detector
    detector = HandPoseDetector()
    
    try:
        # Test with a sample image or camera
        print("[INFO]: Hand pose detector initialized successfully")
        print("[INFO]: Use with camera_stream.py to detect hands in real-time")
        
    except Exception as e:
        print(f"[ERROR]: Failed to initialize hand pose detector: {e}")
    finally:
        detector.cleanup()
