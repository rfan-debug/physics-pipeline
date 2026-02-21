import sys
import os

# Add src to path
# Assuming this script is run from project root or examples/
# We add both possible paths to cover running from root or examples/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '../vla_synthesis/src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Also try adding vla_synthesis/src if run from root and examples is a subdir
root_src_path = os.path.join(os.getcwd(), 'vla_synthesis/src')
if root_src_path not in sys.path:
    sys.path.append(root_src_path)

try:
    from scene_manager import SceneManager
except ImportError:
    # If standard import fails, try relative (less robust but helpful)
    try:
        from vla_synthesis.src.scene_manager import SceneManager
    except ImportError:
        print("Could not import SceneManager. Please ensure PYTHONPATH includes 'vla_synthesis/src'.")
        sys.exit(1)

def main():
    print("Example 1: Basic Scene Loading")
    try:
        # Initialize SceneManager
        # debug=True would show the viewer window (requires GUI environment)
        scene = SceneManager(debug=False)
        print("SceneManager initialized.")

        # Load Robot
        scene.load_robot()
        print("Robot loaded.")

        # Reset Scene (sets up camera and lights)
        scene.reset()
        print("Scene reset (camera and lights randomized).")

        # Render a frame
        rgb, depth, seg = scene.render()
        print(f"Rendered frame with shape: {rgb.shape}, Depth shape: {depth.shape}")

        # In a real environment with PIL installed, you could save the image:
        # from PIL import Image
        # Image.fromarray(rgb).save("output_01.png")
        print("Basic scene test passed.")

    except ImportError as e:
        print(f"Genesis library not found or failed to load: {e}")
        print("This example requires 'genesis' to be installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
