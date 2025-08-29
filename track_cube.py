"""
This script demonstrates AprilTag tracking and robot arm control using differential inverse kinematics.

It uses a webcam to track an AprilTag, translates the 3D pose information into a cube in sim,
and an absolute inverse kinematics controller to control the Franka's end-effector.

Features:
- Real-time AprilTag detection from webcam feed
- Coordinate transformation from camera to simulation world
- Robot arm control to follow AprilTag movement
- Visual markers showing current and goal positions
"""

import argparse
import cv2
import numpy as np
import torch
import time
from isaaclab.app import AppLauncher
from apriltag_stream import AprilTagStream

parser = argparse.ArgumentParser()
parser.add_argument("--tag_size", type=float, default=0.07, help="Size of the AprilTag in meters.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import load_cfg_from_registry

def transform_apriltag_to_world(t_cam_tag, R_cam_tag=None):

    t_cam = np.asarray(t_cam_tag).reshape(3)

    # Scale factor to convert from meters to simulation units
    scale = 1.0
    
    origin = [2.0, 0.5, 0.0]

    world_x = -t_cam[2] * scale + origin[0]
    world_y = t_cam[0] * scale + origin[1]
    world_z = -t_cam[1] * scale + origin[2]

    world_pos = [world_x, world_y, world_z]

    # if R_cam_tag is not None:
    #     try:
    #         from scipy.spatial.transform import Rotation as R
    #     except Exception:
    #         return world_pos

    #     T_cam_to_world = np.array([
    #         [0, 0, 1],   # camera Z -> world X
    #         [-1, 0, 0],  # camera X -> world -Y
    #         [0, -1, 0]   # camera Y -> world -Z
    #     ])

    #     R_world_tag = T_cam_to_world @ R_cam_tag
    #     r = R.from_matrix(R_world_tag)
    #     quat_xyzw = r.as_quat()  # [x, y, z, w]
    #     world_rot = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]

    #     return world_pos, world_rot

    return world_pos

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, tag_size: float):
    robot = scene["robot"]
    cube_object = scene["object"]
    
    # Initialize AprilTag stream
    apriltag_stream = AprilTagStream(cam_index=1, tag_size=tag_size, calib_path="./calibration/calib.npz")
    apriltag_stream.start()
    print("[INFO]: AprilTag stream started. Press 'q' in the webcam window to stop.")

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define initial goal for the arm
    ee_goal = torch.tensor([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0], device=sim.device)

    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    # reset actions
    ik_commands[:] = ee_goal
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # reset controller
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)
    
    # Simulation loop
    while simulation_app.is_running():
        # Display AprilTag webcam feed
        if not apriltag_stream.show_video("AprilTag Webcam Feed"):
            print("[INFO]: Webcam window closed. Stopping simulation.")
            break
            
        # Get latest AprilTag data
        apriltag_data = apriltag_stream.get_latest()
        if apriltag_data is not None:
            # Extract translation (and rotation if available) from AprilTag
            t_cam_tag = apriltag_data["t_cam_tag"]
            R_cam_tag = apriltag_data.get("R_cam_tag", None)

            # Transform to world coordinates
            world_result = transform_apriltag_to_world(t_cam_tag, R_cam_tag)
            world_pos = world_result

            # Keep Z fixed at starting height; update only X and Y from AprilTag
            world_pos_tensor = torch.tensor(world_pos, device=sim.device, dtype=torch.float32)
            new_cube_pos = torch.stack(
                [world_pos_tensor[0], world_pos_tensor[1], torch.tensor(0.0, device=sim.device)]
            )

            # Update the cube object position directly in the scene
            cube_object.write_root_pose_to_sim(
                torch.tensor([new_cube_pos[0], new_cube_pos[1], new_cube_pos[2], 1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0)
            )

            # Update the end-effector goal to follow the cube (0.15m above)
            ee_goal[:3] = new_cube_pos + torch.tensor([0.0, 0.0, 0.15], device=sim.device)
            ik_commands[:] = ee_goal

            # Update controller target with the latest command
            diff_ik_controller.set_command(ik_commands)

            # Print positions every 30 frames
            if count % 30 == 0:
                t = np.asarray(t_cam_tag).reshape(-1)
                print(f"AprilTag (camera) t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
                print(f"Cube (world): [{new_cube_pos[0]:.3f}, {new_cube_pos[1]:.3f}, {new_cube_pos[2]:.3f}]")
        
        # continuously compute IK
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()

        sim.step()
        count += 1
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        
        # Small delay to allow webcam processing
        time.sleep(0.01)
    
    apriltag_stream.stop()
    cv2.destroyAllWindows()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Import environment
    env_cfg = load_cfg_from_registry("Isaac-Lift-Cube-Franka-IK-Abs-v0", "env_cfg_entry_point")
    scene_cfg = env_cfg.scene
    scene_cfg.num_envs = 1
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: Setup complete...")
    print("[INFO]: AprilTag integration enabled. The cube will follow your AprilTag movement.")
    print(f"[INFO]: Using AprilTag size: {args_cli.tag_size} meters")

    run_simulator(sim, scene, args_cli.tag_size)


if __name__ == "__main__":
    main()
    simulation_app.close()