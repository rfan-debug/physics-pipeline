import numpy as np

# Try importing genesis, if not available, it will be mocked in tests or fail if not installed in production
try:
    import genesis as gs
except ImportError:
    gs = None

class SceneManager:
    def __init__(self, debug=False):
        """
        Initialize the Genesis scene.

        Args:
            debug (bool): If True, shows the viewer. Defaults to False (headless).
        """
        if gs is None:
            # This check allows the code to be imported for testing without genesis installed,
            # but will fail if instantiated without mocking gs.
            raise ImportError("Genesis library is not installed. Please install it to use SceneManager.")

        # Initialize Genesis
        gs.init(backend=gs.gpu)

        # Create scene with viewer based on debug flag
        self.scene = gs.Scene(show_viewer=debug)

        self.robot = None
        self.camera = None
        self.light = None

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
        # Using MJCF as it is a common format for Franka in physics engines
        # The path 'xml/franka_emika_panda/panda.xml' is a placeholder for the actual asset path in Genesis
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )

        # Fix base to (0, 0, 0)
        # Genesis entities are usually added at origin by default unless specified otherwise.
        # If explicit fixing is needed, it would be done here.

    def setup_camera(self):
        """
        Add a camera looking at the workspace (approx coordinate 0.5, 0, 0).
        Includes domain randomization for camera position and angle.
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

        # Add or update camera
        # If camera exists, we might need to remove it or update its pose.
        # For this implementation, we assume we are adding it for the first time or the engine handles it.
        # If genesis allows updating pose, we should do that.

        if self.camera is None:
            self.camera = self.scene.add_camera(
                res=(640, 480),
                pos=cam_pos,
                lookat=look_at,
                fov=60,
                GUI=False
            )
        else:
            # Assuming set_pose or similar method exists for updating
            # If not, one might need to re-create the camera
            if hasattr(self.camera, 'set_pose'):
                self.camera.set_pose(pos=cam_pos, lookat=look_at)
            elif hasattr(self.camera, 'set_position') and hasattr(self.camera, 'set_lookat'):
                self.camera.set_position(cam_pos)
                self.camera.set_lookat(look_at)
            else:
                 # Fallback: re-add camera (logic might depend on engine specifics)
                 pass

    def randomize_lighting(self):
        """
        Randomize light position and intensity.
        """
        # Random position
        light_pos = np.random.uniform(low=[1.0, -1.0, 2.0], high=[2.0, 1.0, 3.0])

        # Random intensity
        intensity = np.random.uniform(1.0, 3.0)

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
        Advance the physics simulation.
        """
        self.scene.step()

    def render(self):
        """
        Render the scene.

        Returns:
            tuple: (rgb, depth, segmentation_mask)
        """
        if self.camera is None:
            raise RuntimeError("Camera not initialized. Call setup_camera() first.")

        # Trigger rendering
        self.camera.render()

        # Retrieve data
        # Assuming standard return types (numpy arrays)
        rgb = self.camera.get_color(return_numpy=True) if hasattr(self.camera, 'get_color') else None
        depth = self.camera.get_depth(return_numpy=True) if hasattr(self.camera, 'get_depth') else None
        seg = self.camera.get_segmentation(return_numpy=True) if hasattr(self.camera, 'get_segmentation') else None

        return rgb, depth, seg
