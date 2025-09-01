#!/usr/bin/env python3
"""
Calibrate a personalized palm template from ONE image where the palm is flat on the same plane as an AprilTag.
Outputs a minimal JSON you can use later to scale MediaPipe 3D and/or build objectPoints for solvePnP.
"""

import os, sys, json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pupil_apriltags import Detector

# MediaPipe landmark indices (21-point model)
WRIST = 0
THUMB_CMC = 1
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
PALM_IDX = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

def load_intrinsics(path="./webcam/calib.npz"):
    data = np.load(path)
    return data["mtx"], data["dist"]

def undistort(img, K, D):
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0)
    return cv2.undistort(img, K, D, None, newK)

def detect_tag_corners(img_undist):
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    det = Detector()  # Families auto; tune if needed: Detector(families="tag36h11")
    d = det.detect(gray)
    if not d:
        raise RuntimeError("No AprilTag detected.")
    # pick the largest (most confident) tag
    d.sort(key=lambda x: cv2.contourArea(x.corners.astype(np.float32)), reverse=True)
    return d[0].corners.astype(np.float32)  # shape (4,2)

def order_corners_tl_tr_br_bl(pts):
    # robust reordering
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = np.argmin(s); br = np.argmax(s)
    tr = np.argmin(d); bl = np.argmax(d)
    return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

def hand_keypoints_px(img_undist, hand_landmarker):
    rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = hand_landmarker.detect(mp_img)
    if not res.hand_landmarks:
        raise RuntimeError("No hand detected.")
    # take the first detected hand
    h, w = img_undist.shape[:2]
    pts = []
    for lm in res.hand_landmarks[0]:
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)  # shape (21,2)

def warp_points_to_metric(H, pts_px):
    """Apply homography to 2D pixel points -> metric tag plane coordinates (meters)."""
    pts = cv2.perspectiveTransform(pts_px[None, :, :], H)[0]
    return pts  # (N,2) in meters on plane z=0

def build_palm_template_m(pts_m):
    """Return dict of palm points in meters, expressed in a local palm frame with WRIST at origin."""
    names = ["wrist","thumb_cmc","index_mcp","middle_mcp","ring_mcp","pinky_mcp"]
    # pts_m already contains only the 6 palm points in order [0,1,5,9,13,17]
    # so we can just enumerate directly
    sel = {names[i]: pts_m[i] for i in range(len(pts_m))}
    wrist_xy = sel["wrist"]
    # shift origin to wrist, z=0 for all
    P = {k: [float(v[0] - wrist_xy[0]), float(v[1] - wrist_xy[1]), 0.0] for k,v in sel.items()}
    return P

def visualize_detections(img_undist, tag_corners, hand_kps, palm_kps):
    """Visualize AprilTag corners and hand landmarks on the image."""
    vis_img = img_undist.copy()
    
    # Draw AprilTag corners
    corners_int = tag_corners.astype(np.int32)
    cv2.polylines(vis_img, [corners_int], True, (0, 255, 0), 3)  # Green rectangle
    
    # Label tag corners
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    for i, (corner, label) in enumerate(zip(corners_int, corner_labels)):
        cv2.putText(vis_img, label, (corner[0] + 10, corner[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(vis_img, tuple(corner), 5, (0, 255, 0), -1)
    
    # Draw all hand landmarks (small blue dots)
    for kp in hand_kps:
        cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 2, (255, 0, 0), -1)
    
    # Draw palm landmarks (larger red dots)
    palm_names = ['Wrist', 'Thumb CMC', 'Index MCP', 'Middle MCP', 'Ring MCP', 'Pinky MCP']
    for i, (kp, name) in enumerate(zip(palm_kps, palm_names)):
        cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 8, (0, 0, 255), -1)
        cv2.putText(vis_img, name, (int(kp[0]) + 15, int(kp[1]) - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw connections between palm points
    palm_connections = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]  # wrist to each finger
    for start_idx, end_idx in palm_connections:
        start_pt = palm_kps[start_idx].astype(int)
        end_pt = palm_kps[end_idx].astype(int)
        cv2.line(vis_img, tuple(start_pt), tuple(end_pt), (255, 255, 0), 2)
    
    return vis_img

def main():
    if len(sys.argv) < 3:
        print("Usage: python hand_measurement.py <image_path> <tag_size_meters>")
        sys.exit(1)

    image_path = sys.argv[1]
    tag_size = float(sys.argv[2])

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # 1) Load data
    K, D = load_intrinsics("./webcam/calib.npz")
    img = cv2.imread(image_path); assert img is not None, "Failed to read image"
    img_u = undistort(img, K, D)

    # 2) Detect tag and compute homography to metric square
    corners = detect_tag_corners(img_u)                           # (4,2) px
    corners = order_corners_tl_tr_br_bl(corners)
    dst_metric = np.array([[0,0],[tag_size,0],[tag_size,tag_size],[0,tag_size]], dtype=np.float32)  # meters
    H, _ = cv2.findHomography(corners, dst_metric, method=0)     # img_px -> tag_plane_m

    # 3) Detect hand and pick PALM landmarks
    base = python.BaseOptions(model_asset_path="../models/hand_landmarker.task")
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1, min_hand_detection_confidence=0.5)
    hand_detector = vision.HandLandmarker.create_from_options(opts)
    kps_px = hand_keypoints_px(img_u, hand_detector)              # (21,2) px
    palm_px = kps_px[PALM_IDX]                                   # (6,2) px

    # 4) Visualize detections
    vis_img = visualize_detections(img_u, corners, kps_px, palm_px)
    
    # Display image
    cv2.namedWindow("Tag and Hand Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Tag and Hand Detection", vis_img)
    print("Close the window when done viewing...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5) Warp palm points to metric tag plane (meters)
    palm_m = warp_points_to_metric(H, palm_px)                    # (6,2) m, z=0

    # 6) Compute key measurements (meters)
    wrist_m   = palm_m[0]
    thumb_cmc = palm_m[1]
    idx_mcp   = palm_m[2]
    mid_mcp   = palm_m[3]
    rng_mcp   = palm_m[4]
    pky_mcp   = palm_m[5]

    palm_width_m   = np.linalg.norm(idx_mcp - pky_mcp)
    palm_length_m  = np.linalg.norm(wrist_m - mid_mcp)
    thumb_index_m  = np.linalg.norm(thumb_cmc - idx_mcp)

    # 7) Build palm template (object points) in a local palm frame
    palm_template_m = build_palm_template_m(palm_m)

    # 8) Minimal JSON
    out = {
        "source_image": os.path.basename(image_path),
        "tag_size_m": float(tag_size),
        "landmark_ids": {
            "wrist": WRIST,
            "thumb_cmc": THUMB_CMC,
            "index_mcp": INDEX_MCP,
            "middle_mcp": MIDDLE_MCP,
            "ring_mcp": RING_MCP,
            "pinky_mcp": PINKY_MCP
        },
        "measurements_mm": {
            "palm_width_mm": float(palm_width_m * 1000.0),
            "palm_length_mm": float(palm_length_m * 1000.0),
            "thumb_index_base_mm": float(thumb_index_m * 1000.0)
        },
        "palm_template_m": palm_template_m  # use as objectPoints (z=0) for planar PnP
    }

    out_path = "./hand/measurements.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    print(json.dumps(out["measurements_mm"], indent=2))

if __name__ == "__main__":
    main()
