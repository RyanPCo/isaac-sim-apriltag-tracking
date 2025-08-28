## AprilTag Robot Control in Isaac Sim

This project implements real-time AprilTag detection to control the Franka Panda arm in NVIDIA Isaac Sim. It uses a webcam to track an AprilTag, translates the 3D pose information into a cube in sim, and an absolute inverse kinematics controller to control the Franka's end-effector.

![Demo](./assets/demo.gif)

## Technical Architecture

1. **AprilTagStream Class** (`apriltag_stream.py`)
   - Manages webcam input and AprilTag detection
   - Handles camera calibration loading and undistortion
   - Estimates 3D pose (rotation and translation) of detected tags

2. **Robot Control in Simulation** (`track_cube.py`)
   - Integrates AprilTag data in Isaac Sim
   - Uses [Isaac-Lift-Cube-Franka-IK-Abs-v0 environment](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_abs_env_cfg.py)
   - Maps camera coordinates to cube in simulation world coordinates
   - Uses absolute inverse kinematics to control Franka arm

## Installation

1. **Activate Isaac Lab Environment**

This varies by installation method, follow [NVIDIA's official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Camera Calibration**
```bash
python camera_calibration.py
```
- Place an 8x6 chessboard pattern in view
- Capture multiple images from different angles (recommended ~20)
- Calibration data saved to `./calibration/calib.npz`

2. **Full Simulation**
```bash
python track_cube.py --tag_size=[TAG_SIZE]
```
- Launches Isaac Sim with robot control, make sure to specify tag_size
- Robot arm tracks AprilTag movement using IK controller
