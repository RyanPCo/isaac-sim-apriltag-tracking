# apriltag_stream.py
import cv2, threading, time, numpy as np
from pupil_apriltags import Detector

class AprilTagStream:
    def __init__(self, cam_index=0, width=1920, height=1080, tag_size=0.07, calib_path="./calibration/webcam/calib.npz"):
        self.cap = cv2.VideoCapture(cam_index)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.detector = Detector(families="tag36h11", nthreads=1, refine_edges=True)

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
        self.tag_size = float(tag_size)
        self.dist = dist
        self._undistort_enabled = np.any(np.abs(self.dist) > 1e-12)
        self._undistort_initialized = False
        self._map1 = None
        self._map2 = None
        self._K_rectified = self.K.copy()
        self.lock = threading.Lock()
        self.latest = None
        self.running = False
        self.thread = None
        self._latest_frame_vis = None

    def _draw_detections(self, frame, dets, R_cam_tag, t_cam_tag):
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
        if R_cam_tag is not None and t_cam_tag is not None:
            rvec, _ = cv2.Rodrigues(R_cam_tag)
            axis_len = self.tag_size * 1.5
            axes_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1,3)
            imgpts, _ = cv2.projectPoints(axes_3d, rvec, t_cam_tag.reshape(3,1), self.K, self.dist)
            imgpts = imgpts.reshape(-1,2).astype(int)
            origin = tuple(imgpts[0])
            cv2.line(vis, origin, tuple(imgpts[1]), (0,0,255), 3)   # X - red
            cv2.line(vis, origin, tuple(imgpts[2]), (0,255,0), 3)   # Y - green
            cv2.line(vis, origin, tuple(imgpts[3]), (255,0,0), 3)   # Z - blue
            t = t_cam_tag.reshape(-1)
            cv2.putText(vis, f"t:[{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}] m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return vis

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok: 
                time.sleep(0.01); continue
            # Lazy initialize undistortion maps when we know frame size
            if self._undistort_enabled and not self._undistort_initialized:
                h, w = frame.shape[:2]
                newK, _roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
                self._K_rectified = newK.astype(np.float32)
                self._map1, self._map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
                self._undistort_initialized = True

            # Use rectified frame for detection if calibration available
            if self._undistort_enabled and self._undistort_initialized:
                frame_for_detect = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
                K_for_detect = self._K_rectified
            else:
                frame_for_detect = frame
                K_for_detect = self.K
            gray = cv2.cvtColor(frame_for_detect, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detect(gray, estimate_tag_pose=True,
                                        camera_params=(K_for_detect[0,0], K_for_detect[1,1], K_for_detect[0,2], K_for_detect[1,2]),
                                        tag_size=self.tag_size)
            R_cam_tag = None
            t_cam_tag = None
            if dets:
                d = dets[0]
                R_cam_tag = np.array(d.pose_R, dtype=np.float32)
                t_cam_tag = np.array(d.pose_t, dtype=np.float32).reshape(3,1)
                with self.lock:
                    self.latest = {"R_cam_tag": R_cam_tag, "t_cam_tag": t_cam_tag, "ts": time.time()}
            if self._undistort_enabled and self._undistort_initialized:
                old_K, old_dist = self.K, self.dist
                self.K = self._K_rectified
                self.dist = np.zeros(5, dtype=np.float32)
                vis = self._draw_detections(frame_for_detect, dets, R_cam_tag, t_cam_tag)
                self.K, self.dist = old_K, old_dist
            else:
                vis = self._draw_detections(frame, dets, R_cam_tag, t_cam_tag)
            with self.lock:
                self._latest_frame_vis = vis
            # small sleep to reduce CPU
            time.sleep(0.005)

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.cap: self.cap.release()

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else {k: v.copy() if hasattr(v, "copy") else v for k, v in self.latest.items()}

    def get_latest_frame(self):
        with self.lock:
            if self._latest_frame_vis is None:
                return None
            return self._latest_frame_vis.copy()

    def show_video(self, window_name="AprilTag Detection"):
        try:
            # Make window resizable
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            frame = self.get_latest_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
            return True
        except Exception as e:
            print(f"[WARNING]: Error displaying AprilTag video: {e}")
            return True


if __name__ == "__main__":
    stream = AprilTagStream(cam_index=1, width=1920, height=1080, tag_size=0.07, calib_path="./calibration/webcam/calib.npz")
    stream.start()
    last_ts_printed = None
    try:
        while True:
            # Use show_video method which handles resizable window
            if not stream.show_video("AprilTag Stream"):
                break
            latest = stream.get_latest()
            if latest is not None:
                ts = latest.get("ts", None)
                if ts is not None and ts != last_ts_printed:
                    R = latest["R_cam_tag"]
                    t = latest["t_cam_tag"].reshape(-1)
                    print(f"ts={ts:.3f} R=\n{R}\n t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
                    last_ts_printed = ts
            time.sleep(0.005)
    finally:
        stream.stop()
        cv2.destroyAllWindows()
