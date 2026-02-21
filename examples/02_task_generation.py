import sys
import os
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# If run from project root, use vla_synthesis/src
# If run from examples/, use ../vla_synthesis/src
# This allows running from both places
if os.path.exists(os.path.join(current_dir, '../vla_synthesis/src')):
    sys.path.append(os.path.join(current_dir, '../vla_synthesis/src'))
elif os.path.exists(os.path.join(current_dir, 'vla_synthesis/src')): # run from root
    sys.path.append(os.path.join(current_dir, 'vla_synthesis/src'))
else:
    # Try assuming current_dir IS root/examples so project root is ..
    sys.path.append(os.path.join(current_dir, '..', 'vla_synthesis', 'src'))

try:
    from scene_manager import SceneManager
    from task_generator import TaskGenerator
except ImportError:
    print("Could not import modules. Ensure PYTHONPATH includes 'vla_synthesis/src'.")
    sys.exit(1)

def main():
    print("Example 2: Task Generation")
    try:
        # Initialize
        scene = SceneManager(debug=False)
        scene.load_robot()
        task_gen = TaskGenerator()
        print("Components initialized.")

        for i in range(3):
            print(f"\n--- Iteration {i+1} ---")

            # Reset scene (randomize camera/lighting)
            scene.reset()

            # Generate task (randomize object and create instruction)
            instruction, target_obj = task_gen.reset_task(scene.scene)
            print(f"Instruction: {instruction}")

            if target_obj:
                print("Target object created successfully.")

                # Render to verify scene state
                rgb, _, _ = scene.render()
                print(f"Scene rendered. Frame shape: {rgb.shape}")

                # Simulate waiting for user to read/inspect
                # time.sleep(1)
            else:
                print("Failed to create target object.")

        print("\nTask generation test passed.")

    except ImportError as e:
        print(f"Genesis library not found: {e}")
        print("This example requires 'genesis' to be installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
