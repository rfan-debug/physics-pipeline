import sys
import os
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check for common locations
paths_to_check = [
    os.path.join(current_dir, '../vla_synthesis/src'), # Run from examples/
    os.path.join(current_dir, 'vla_synthesis/src'),    # Run from root/
    os.path.join(current_dir, '../src'),               # If structure differs
]

for p in paths_to_check:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    from scene_manager import SceneManager
    from task_generator import TaskGenerator
    from planner import SimplePlanner
    from recorder import HDF5Recorder
except ImportError:
    # Try importing from package
    try:
        from vla_synthesis.src.scene_manager import SceneManager
        from vla_synthesis.src.task_generator import TaskGenerator
        from vla_synthesis.src.planner import SimplePlanner
        from vla_synthesis.src.recorder import HDF5Recorder
    except ImportError:
        print("Could not import vla_synthesis modules. Please check PYTHONPATH.")
        sys.exit(1)

def main():
    print("Example 3: Full Episode Simulation")
    try:
        # Initialize Scene
        scene = SceneManager(debug=False)
        scene.load_robot()
        print("Scene initialized.")

        # Initialize Components
        task_gen = TaskGenerator()
        planner = SimplePlanner()
        print("Planner and TaskGenerator initialized.")

        # Episode Loop (single episode for demo)
        episode_idx = 0
        output_file = "example_episode.h5"

        # Using context manager for recorder
        with HDF5Recorder(output_file) as recorder:
            print(f"Recorder initialized, saving to {output_file}")

            # Reset Environment
            scene.reset()
            instr, target_obj = task_gen.reset_task(scene.scene)
            print(f"Instruction: {instr}")

            if not target_obj:
                print("Failed to generate task object.")
                return

            # Get target position
            # Handle potential API differences
            if hasattr(target_obj, 'get_pos'):
                 target_pos = np.array(target_obj.get_pos())
            elif hasattr(target_obj, 'get_position'):
                 target_pos = np.array(target_obj.get_position())
            else:
                 # Fallback for testing without real physics
                 print("Warning: target_obj has no position method. Using mock position.")
                 target_pos = np.array([0.5, 0.0, 0.05])

            print(f"Target Position: {target_pos}")

            # Plan Trajectory
            # This might fail if IK solver is not available (mocked genesis)
            try:
                actions = planner.plan_grasp(scene.robot, target_pos)
                print(f"Trajectory planned: {len(actions)} steps.")
            except Exception as e:
                print(f"Planning failed: {e}")
                # Create a mock trajectory for demonstration purposes if planning fails
                print("Using mock trajectory for demonstration.")
                actions = [np.zeros(9) for _ in range(50)]

            # Execute and Record
            recorder.create_episode_group(episode_idx)

            step_count = 0
            for action in actions:
                # Log progress
                if step_count % 20 == 0:
                    print(f"Executing step {step_count}/{len(actions)}")

                # Apply action
                if hasattr(scene.robot, 'control_dofs_position'):
                    scene.robot.control_dofs_position(action)
                elif hasattr(scene.robot, 'control_joints'):
                    scene.robot.control_joints(action)
                elif hasattr(scene.robot, 'set_q'):
                     scene.robot.set_q(action)

                # Step physics
                scene.step()

                # Render
                rgb, _, _ = scene.render()

                # Save
                recorder.save_step(
                    episode_idx=episode_idx,
                    observation=rgb,
                    action=action,
                    instruction=instr,
                    reward=0.0 # Placeholder
                )
                step_count += 1

            print(f"Episode {episode_idx} completed.")

        print("Full episode simulation finished successfully.")

    except ImportError as e:
        print(f"Genesis library not found: {e}")
        print("This example requires 'genesis' to be installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
