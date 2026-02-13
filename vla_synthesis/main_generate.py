from vla_synthesis.src.scene_manager import SceneManager
from vla_synthesis.src.task_generator import TaskGenerator
from vla_synthesis.src.planner import SimplePlanner
from vla_synthesis.src.recorder import HDF5Recorder
import numpy as np

def main():
    # Configuration
    NUM_EPISODES = 1000
    MAX_STEPS = 200
    SAVE_PATH = "vla_dataset.h5"

    print("Initializing components...")
    try:
        scene = SceneManager()
        task_gen = TaskGenerator()
        planner = SimplePlanner()
        recorder = HDF5Recorder(SAVE_PATH)
    except ImportError as e:
        print(f"Error initializing components: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        return

    # Load robot once
    try:
        scene.load_robot()
    except Exception as e:
        print(f"Failed to load robot: {e}")
        return

    print(f"Starting generation of {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        try:
            # 1. Reset Environment
            # This clears previous objects and adds a new one
            text_instr, target_obj = task_gen.reset_task(scene)
            scene.randomize_lighting()
            scene.setup_camera() # Ensure camera is set/randomized

            # 2. Plan Expert Trajectory
            # Get target position. Genesis entities usually have get_pos() or we might need to access pos attribute
            if hasattr(target_obj, 'get_pos'):
                target_pos = target_obj.get_pos()
            elif hasattr(target_obj, 'pos'):
                 target_pos = target_obj.pos
            else:
                 # Fallback/Assume mock behavior or check docs (which we can't)
                 # We'll try generic access or default to center if missing (should not happen with valid entities)
                 print(f"Warning: Could not determine position of target object {target_obj}. Skipping.")
                 continue

            try:
                actions_trajectory = planner.plan_grasp(scene.robot, target_pos)
            except Exception as ik_error:
                print(f"Episode {ep}: Planning/IK failed ({ik_error}). Skipping.")
                continue

            if not actions_trajectory:
                print(f"Episode {ep}: Planner returned empty trajectory. Skipping.")
                continue

            # 3. Execute and Record
            recorder.create_episode_group(ep)

            # Limit steps if trajectory is too long
            trajectory = actions_trajectory[:MAX_STEPS]

            for i, action in enumerate(trajectory):
                # Control robot
                # Try standard control methods
                if hasattr(scene.robot, 'control_joints'):
                    scene.robot.control_joints(action)
                elif hasattr(scene.robot, 'control_dofs_position'):
                    scene.robot.control_dofs_position(action)
                else:
                     # Mock or fail
                     pass

                # Step physics
                scene.step()

                # Render
                rgb, depth, mask = scene.render()

                # Get current end-effector state if possible (optional)
                state = None
                if hasattr(scene.robot, 'get_ee_pose'):
                    state = np.array(scene.robot.get_ee_pose()) # simplified

                # Save step
                recorder.save_step(
                    episode_idx=ep,
                    observation=rgb,
                    action=action,
                    instruction=text_instr,
                    reward=0.0, # Placeholder reward
                    state=state
                )

            print(f"Episode {ep} generated: {text_instr}")

        except Exception as e:
            print(f"Episode {ep}: Unexpected error: {e}")
            continue

    recorder.close()
    print("Data generation complete.")

if __name__ == "__main__":
    main()
