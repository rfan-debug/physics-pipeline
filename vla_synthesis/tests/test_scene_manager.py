import unittest
import sys
import numpy as np
from unittest.mock import MagicMock, patch
import os

# Mock genesis module before importing SceneManager
sys.modules['genesis'] = MagicMock()
import genesis as gs

# Ensure we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vla_synthesis.src.scene_manager import SceneManager

class TestSceneManager(unittest.TestCase):
    def setUp(self):
        # Reset the mock for each test
        gs.reset_mock()
        # Mock specific attributes
        gs.morphs = MagicMock()
        gs.gpu = 'gpu'

        # Setup Scene mock
        self.mock_scene = MagicMock()
        gs.Scene.return_value = self.mock_scene

        # Instantiate SceneManager
        self.manager = SceneManager(debug=False)

    def test_init(self):
        """Test initialization of SceneManager."""
        gs.init.assert_called_with(backend=gs.gpu)
        gs.Scene.assert_called_with(show_viewer=False)
        self.assertEqual(self.manager.scene, self.mock_scene)

    def test_load_robot_franka(self):
        """Test loading robot with Franka asset."""
        # Setup gs.morphs.Franka
        gs.morphs.Franka = MagicMock()

        self.manager.load_robot()

        # Check Plane creation
        gs.morphs.Plane.assert_called()
        # Check that add_entity was called with the result of Plane()
        # Note: We can't easily match the exact object instance unless we mock the return of Plane()
        # But we can verify add_entity was called.
        self.assertTrue(self.mock_scene.add_entity.called)

        # Check Franka creation
        gs.morphs.Franka.assert_called_with(fixed=True, pos=(0, 0, 0))
        self.assertIsNotNone(self.manager.robot)

    def test_setup_camera_initial(self):
        """Test initial camera setup."""
        self.manager.setup_camera()

        self.mock_scene.add_camera.assert_called_once()
        self.assertIsNotNone(self.manager.camera)

        # Check call args to verify randomization happened
        call_args = self.mock_scene.add_camera.call_args
        kwargs = call_args[1]

        base_pos = np.array([1.0, 0.0, 0.8])
        look_at_target = np.array([0.5, 0.0, 0.0])

        # Pos and lookat should be close but not exact due to randomization
        np.testing.assert_allclose(kwargs['pos'], base_pos, atol=0.06)
        np.testing.assert_allclose(kwargs['lookat'], look_at_target, atol=0.03)

    def test_setup_camera_update(self):
        """Test updating camera on subsequent calls."""
        # First call creates camera
        self.manager.setup_camera()
        mock_cam = self.manager.camera
        mock_cam.set_pose = MagicMock()

        # Second call should update
        self.manager.setup_camera()

        # Should not create new camera (add_camera called once total)
        self.mock_scene.add_camera.assert_called_once()
        # Should call set_pose on the existing camera
        mock_cam.set_pose.assert_called_once()

    def test_randomize_lighting(self):
        """Test lighting randomization."""
        self.manager.randomize_lighting()

        self.mock_scene.add_entity.assert_called()
        gs.morphs.Light.assert_called()
        self.assertIsNotNone(self.manager.light)

    def test_step(self):
        """Test simulation step."""
        self.manager.step()
        self.mock_scene.step.assert_called_once()

    def test_render(self):
        """Test rendering."""
        # Setup mock camera
        mock_cam = MagicMock()
        self.manager.camera = mock_cam

        # Setup return values for camera methods
        mock_cam.get_color.return_value = np.zeros((480, 640, 3))
        mock_cam.get_depth.return_value = np.zeros((480, 640))
        mock_cam.get_segmentation.return_value = np.zeros((480, 640))

        rgb, depth, seg = self.manager.render()

        mock_cam.render.assert_called_once()
        mock_cam.get_color.assert_called()
        mock_cam.get_depth.assert_called()
        mock_cam.get_segmentation.assert_called()

        self.assertIsInstance(rgb, np.ndarray)

    def test_reset(self):
        """Test reset calls randomization methods."""
        with patch.object(self.manager, 'setup_camera') as mock_cam_setup, \
             patch.object(self.manager, 'randomize_lighting') as mock_light_setup:

            self.manager.reset()

            mock_cam_setup.assert_called_once()
            mock_light_setup.assert_called_once()

    def test_render_no_camera(self):
        """Test rendering when camera is not initialized."""
        # Ensure camera is None
        self.manager.camera = None

        rgb, depth, seg = self.manager.render()

        # Check return types and shapes
        self.assertIsInstance(rgb, np.ndarray)
        self.assertIsInstance(depth, np.ndarray)
        self.assertIsInstance(seg, np.ndarray)

        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(depth.shape, (480, 640))
        self.assertEqual(seg.shape, (480, 640))

        # Check dtype
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(seg.dtype, np.int32)

if __name__ == '__main__':
    unittest.main()
