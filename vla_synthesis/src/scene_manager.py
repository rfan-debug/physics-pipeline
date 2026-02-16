import numpy as np
import warnings
from typing import Tuple, Optional

# Try importing genesis, if not available, it will be mocked in tests or fail if not installed in production
try:
    import genesis as gs
except ImportError:
    gs = None

class SceneManager:
    """
    Manages the Genesis scene for VLA data synthesis.

    This class handles:
    - Initializing the physics engine and scene.
    - Loading the robot (Franka Panda) and workspace.
    - Setting up the camera with domain randomization (position and angle).
    - Setting up lighting with domain randomization (position and intensity).
    - Stepping the simulation.
    - Rendering observations (RGB, Depth, Segmentation).
    """
    def __init__(self, debug: bool = False):
        """
        Initialize the Genesis scene.

        Args:
            debug (bool): If True, shows the viewer. Defaults to False (headless).
        """
        if gs is None:
            raise ImportError("Genesis library is not installed. Please install it to use SceneManager.")

        # Initialize Genesis
        gs.init(backend=gs.gpu)

        # Create scene with viewer based on debug flag
        self.scene = gs.Scene(show_viewer=debug)

        self.robot = None
        self.camera = None
        self.light = None

        self.render_res = (640, 480)

    def load_robot(self):
        """
        Load a Franka Panda robot from Genesis standard assets.
        Fix its base to (0, 0, 0).
        """
        # Create a plane for the robot to stand on
        self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # Load Franka Panda
        # Prioritize standard asset loader for Franka or Panda
        try:
            if hasattr(gs.morphs, 'Franka'):
                self.robot = self.scene.add_entity(
                    gs.morphs.Franka(fixed=True, pos=(0, 0, 0))
                )
            elif hasattr(gs.morphs, 'Panda'):
                 self.robot = self.scene.add_entity(
                    gs.morphs.Panda(fixed=True, pos=(0, 0, 0))
                )
            else:
                # Fallback to MJCF if specific class not found
                warnings.warn("Standard Franka/Panda asset not found in gs.morphs. Trying MJCF with placeholder path.")
                self.robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file='xml/franka_emika_panda/panda.xml',
                        pos=(0, 0, 0),
                        fixed=True,
                    ),
                )
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Failed to load robot asset: {e}") from e

    def setup_camera(self):
        """
        Add or update a camera looking at the workspace (approx coordinate 0.5, 0, 0).
        Includes domain randomization for camera position and angle (perturbations applied every reset).
        """
        # Base camera position (approximate, looking at workspace)
        # Positioned slightly elevated and back to view the workspace at (0.5, 0, 0)
        base_pos = np.array([1.0, 0.0, 0.8])
        look_at_target = np.array([0.5, 0.0, 0.0])

        # Domain randomization: Add small random perturbations
        # Randomize position
        pos_noise = np.random.uniform(-0.05, 0.05, size=3)
        cam_pos = base_pos + pos_noise

        # Randomize look-at target slightly to vary the angle
        target_noise = np.random.uniform(-0.02, 0.02, size=3)
        look_at = look_at_target + target_noise

        # Add or update camera.
        # If camera doesn't exist, create it. Otherwise, update its pose to reflect randomization.
        if self.camera is None:
            self.camera = self.scene.add_camera(
                res=self.render_res,
                pos=cam_pos,
                lookat=look_at,
                fov=60,
                GUI=False
            )
        else:
            # Update existing camera pose
            # Assuming set_pose method exists for updating
            if hasattr(self.camera, 'set_pose'):
                self.camera.set_pose(pos=cam_pos, lookat=look_at)
            # Fallback for different API styles, or re-add logic if needed
            elif hasattr(self.camera, 'set_position') and hasattr(self.camera, 'set_lookat'):
                self.camera.set_position(cam_pos)
                self.camera.set_lookat(look_at)

    def randomize_lighting(self):
        """
        Randomize light position and intensity.
        """
        # Random position
        light_pos = np.random.uniform(low=[1.0, 1.0, 2.0], high=[3.0, 3.0, 4.0])

        # Random intensity
        intensity = np.random.uniform(2.0, 5.0)

        # Add or update light.
        # If light doesn't exist, create it. Otherwise, update its parameters to reflect randomization.
        if self.light is None:
            self.light = self.scene.add_entity(
                gs.morphs.Light(
                    pos=light_pos,
                    color=(1.0, 1.0, 1.0),
                    intensity=intensity
                )
            )
        else:
            # Update existing light if possible
            if hasattr(self.light, 'set_pos'):
                self.light.set_pos(light_pos)
            if hasattr(self.light, 'set_intensity'):
                self.light.set_intensity(intensity)

    def step(self):
        """
        Advance the physics simulation by one step.
        """
        self.scene.step()

    def render(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the scene.

        Returns:
            tuple: (rgb, depth, segmentation_mask)
        """
        width, height = self.render_res

        # Initialize zero arrays for fallback
        zero_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        zero_depth = np.zeros((height, width), dtype=np.float32)
        zero_seg = np.zeros((height, width), dtype=np.int32)

        if self.camera is None:
            return zero_rgb, zero_depth, zero_seg

        # Trigger rendering
        self.camera.render()

        # Retrieve data from camera
        # Assuming standard return types (numpy arrays)
        rgb = self.camera.get_color(return_numpy=True) if hasattr(self.camera, 'get_color') else None
        depth = self.camera.get_depth(return_numpy=True) if hasattr(self.camera, 'get_depth') else None
        seg = self.camera.get_segmentation(return_numpy=True) if hasattr(self.camera, 'get_segmentation') else None

        # Return zeros if retrieval fails (None)
        return (
            rgb if rgb is not None else zero_rgb,
            depth if depth is not None else zero_depth,
            seg if seg is not None else zero_seg
        )

    def reset(self):
        """
        Reset the scene, applying domain randomization to camera and lighting.
        """
        self.setup_camera()
        self.randomize_lighting()
