# apriltag_tracker.py
import cv2, numpy as np, time
from pupil_apriltags import Detector

class AprilTagTracker:
    def __init__(self, tag_size=0.07):
        self.detector = Detector(families="tag36h11", nthreads=1, refine_edges=True)
        self.tag_size = float(tag_size)
        self.latest = None

    def detect_tags(self, frame, camera_params):
        """
        Detect AprilTags in a frame and estimate pose
        
        Args:
            frame: Input frame (grayscale or BGR)
            camera_params: Camera intrinsic matrix K
            
        Returns:
            dict with detection results or None if no tags found
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Extract camera parameters
        fx, fy, cx, cy = camera_params[0,0], camera_params[1,1], camera_params[0,2], camera_params[1,2]
        
        # Detect tags
        dets = self.detector.detect(gray, estimate_tag_pose=True,
                                    camera_params=(fx, fy, cx, cy),
                                    tag_size=self.tag_size)
        
        if not dets:
            return None
            
        # Get pose of first detected tag
        d = dets[0]
        R_cam_tag = np.array(d.pose_R, dtype=np.float32)
        t_cam_tag = np.array(d.pose_t, dtype=np.float32).reshape(3,1)
        
        result = {
            "R_cam_tag": R_cam_tag,
            "t_cam_tag": t_cam_tag,
            "detections": dets,
            "ts": time.time()
        }
        
        self.latest = result
        return result

    def get_latest(self):
        """Get latest detection results"""
        if self.latest is None:
            return None
        return {k: v.copy() if hasattr(v, "copy") else v for k, v in self.latest.items()}

    def draw_detections(self, frame, dets, R_cam_tag=None, t_cam_tag=None, camera_params=None):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame to draw on
            dets: Detection results from detect_tags
            R_cam_tag: Rotation matrix (optional, for pose visualization)
            t_cam_tag: Translation vector (optional, for pose visualization)
            camera_params: Camera intrinsic matrix K (optional, for pose visualization)
        """
        vis = frame.copy()
        
        # Draw tag corners and ID
        for d in dets:
            pts = d.corners.astype(int)
            for i in range(4):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i+1)%4])
                cv2.line(vis, p1, p2, (0, 255, 0), 2)
            c = tuple(np.round(d.center).astype(int))
            cv2.circle(vis, c, 4, (0, 0, 255), -1)
            if hasattr(d, 'tag_id'):
                cv2.putText(vis, f"id:{d.tag_id}", (c[0]+6, c[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        
        # Draw 3D axes for the first detection using pose (if available)
        if R_cam_tag is not None and t_cam_tag is not None and camera_params is not None:
            rvec, _ = cv2.Rodrigues(R_cam_tag)
            axis_len = self.tag_size * 1.5
            axes_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1,3)
            imgpts, _ = cv2.projectPoints(axes_3d, rvec, t_cam_tag.reshape(3,1), camera_params, None)
            imgpts = imgpts.reshape(-1,2).astype(int)
            origin = tuple(imgpts[0])
            cv2.line(vis, origin, tuple(imgpts[1]), (0,0,255), 3)   # X - red
            cv2.line(vis, origin, tuple(imgpts[2]), (0,255,0), 3)   # Y - green
            cv2.line(vis, origin, tuple(imgpts[3]), (255,0,0), 3)   # Z - blue
            t = t_cam_tag.reshape(-1)
            cv2.putText(vis, f"April Tag:[{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}] m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        return vis
