import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Mock genesis before importing anything else
sys.modules['genesis'] = MagicMock()

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Also ensure vla_synthesis/src is in path for main_generate internal imports if needed
# (main_generate handles its own imports but for test imports we need consistency)

from vla_synthesis.src.scene_manager import SceneManager
from vla_synthesis.src.task_generator import TaskGenerator
from vla_synthesis.src.planner import SimplePlanner
from vla_synthesis.src.recorder import HDF5Recorder
import vla_synthesis.main_generate as main_gen

class TestMainIntegration(unittest.TestCase):
    @patch('vla_synthesis.main_generate.SceneManager')
    @patch('vla_synthesis.main_generate.TaskGenerator')
    @patch('vla_synthesis.main_generate.SimplePlanner')
    @patch('vla_synthesis.main_generate.HDF5Recorder')
    def test_main_loop(self, MockRecorder, MockPlanner, MockTaskGen, MockSceneManager):
        # Setup Mocks
        scene_instance = MockSceneManager.return_value
        task_gen_instance = MockTaskGen.return_value
        planner_instance = MockPlanner.return_value
        recorder_instance = MockRecorder.return_value

        # Mock TaskGenerator behavior
        mock_target_obj = MagicMock()
        mock_target_obj.get_pos.return_value = np.array([0.5, 0.0, 0.05])
        task_gen_instance.reset_task.return_value = ("Pick up the cube", mock_target_obj)

        # Mock Planner behavior
        # Return a dummy trajectory of 10 steps
        dummy_traj = [np.zeros(9) for _ in range(10)]
        planner_instance.plan_grasp.return_value = dummy_traj

        # Mock SceneManager render
        scene_instance.render.return_value = (np.zeros((480, 640, 3)), np.zeros((480, 640)), np.zeros((480, 640)))

        # Mock robot control
        scene_instance.robot.control_dofs_position = MagicMock()

        # Run main with reduced episodes for speed
        # Check if we can patch global variable or just run it and break early?
        # Since main() hardcodes NUM_EPISODES = 1000, we can't easily change it without modifying code.
        # But we can mock range to return a short list.
        with patch('builtins.range', return_value=[0]):
            main_gen.main()

        # Assertions
        # 1. Check Initialization
        MockSceneManager.assert_called_once()
        scene_instance.load_robot.assert_called_once()
        MockTaskGen.assert_called_once()
        MockPlanner.assert_called_once()
        MockRecorder.assert_called_with("vla_dataset.h5")

        # 2. Check Loop execution (1 episode)
        scene_instance.reset.assert_called_once()
        task_gen_instance.reset_task.assert_called_once()
        # Ensure correct scene passed (mocked scene property)
        # We can't strictly verify scene.scene is passed if scene.scene is also a mock created on the fly,
        # but we can check call args.

        planner_instance.plan_grasp.assert_called_once()
        recorder_instance.create_episode_group.assert_called_with(0)

        # 3. Check inner loop (actions)
        # 10 steps in trajectory
        self.assertEqual(scene_instance.step.call_count, 10)
        self.assertEqual(scene_instance.render.call_count, 10)
        self.assertEqual(recorder_instance.save_step.call_count, 10)

        # Verify robot control was called
        self.assertEqual(scene_instance.robot.control_dofs_position.call_count, 10)

if __name__ == '__main__':
    unittest.main()
