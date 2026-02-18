import sys
import os
import numpy as np
import warnings

# Ensure src is in path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from scene_manager import SceneManager
from task_generator import TaskGenerator
from planner import SimplePlanner
from recorder import HDF5Recorder

def main():
    # Configuration
    NUM_EPISODES = 1000
    MAX_STEPS = 200

    print("Initializing components...")
    try:
        # Initialize SceneManager
        # We use debug=False for faster generation without GUI
        scene = SceneManager(debug=False)

        # Load the robot and ground plane
        scene.load_robot()

        # Initialize Task Generator and Planner
        task_gen = TaskGenerator()
        planner = SimplePlanner()

        # Initialize Recorder
        recorder = HDF5Recorder("vla_dataset.h5")

    except ImportError as e:
        print(f"Initialization failed (ImportError): {e}")
        print("Please ensure 'genesis' and other dependencies are installed.")
        return
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    print(f"Starting generation of {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        try:
            # 1. Reset Environment
            # Reset scene (randomizes camera and lighting)
            scene.reset()

            # Generate new task
            # We pass the underlying genesis scene object to TaskGenerator
            # reset_task returns instruction and target object entity
            text_instr, target_obj = task_gen.reset_task(scene.scene)

            # Explicitly randomize lighting as per requirements (though reset() does it too)
            scene.randomize_lighting()

            if target_obj is None:
                warnings.warn(f"Episode {ep}: Failed to generate task (target_obj is None). Skipping.")
                continue

            # 2. Plan Expert Trajectory
            # Get target position. Attempt get_position() as per prompt, fallback to get_pos()
            if hasattr(target_obj, 'get_position'):
                target_pos = target_obj.get_position()
            elif hasattr(target_obj, 'get_pos'):
                target_pos = target_obj.get_pos()
            else:
                # Fallback if neither exists (e.g. during mock/testing without real genesis)
                # Assuming [0.5, 0, 0.05] as a safe default for testing
                warnings.warn(f"Episode {ep}: target_obj has no get_position/get_pos. Using default.")
                target_pos = np.array([0.5, 0.0, 0.05])

            # Plan grasp trajectory
            actions_trajectory = planner.plan_grasp(scene.robot, target_pos)

            # 3. Execute and Record
            recorder.create_episode_group(ep)

            step_count = 0
            for action in actions_trajectory:
                if step_count >= MAX_STEPS:
                    break

                # Control robot
                # Use control_dofs_position if available, or control_joints if per prompt
                if hasattr(scene.robot, 'control_dofs_position'):
                    scene.robot.control_dofs_position(action)
                elif hasattr(scene.robot, 'control_joints'):
                    scene.robot.control_joints(action)
                elif hasattr(scene.robot, 'set_q'):
                     scene.robot.set_q(action)

                # Step simulation
                scene.step()

                # Render observations
                rgb, depth, mask = scene.render()

                # Save step data
                recorder.save_step(
                    episode_idx=ep,
                    observation=rgb,
                    action=action,
                    instruction=text_instr,
                    reward=0.0 # Placeholder reward
                )
                step_count += 1

            print(f"Episode {ep} generated: {text_instr}")

        except RuntimeError as e:
            # Catch IK failures or other runtime errors
            print(f"Episode {ep} failed (RuntimeError): {e}")
            continue
        except Exception as e:
            print(f"Episode {ep} failed (Unexpected): {e}")
            continue

    recorder.close()
    print("Data generation complete.")

if __name__ == "__main__":
    main()
