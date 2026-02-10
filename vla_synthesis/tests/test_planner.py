import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Adjust path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vla_synthesis.src.planner import SimplePlanner

class TestSimplePlanner(unittest.TestCase):
    def setUp(self):
        self.planner = SimplePlanner()
        self.robot = MagicMock()
        self.robot.get_q.return_value = np.zeros(9)
        self.robot.inverse_kinematics.return_value = np.ones(9)

    def test_plan_grasp(self):
        target_pos = np.array([0.5, 0.0, 0.05])
        trajectory = self.planner.plan_grasp(self.robot, target_pos)

        # Check if inverse_kinematics was called at least 3 times (pre-grasp, grasp, lift)
        self.assertGreaterEqual(self.robot.inverse_kinematics.call_count, 3)

        # Check the length of the trajectory
        # 50 steps * 3 movements + gripper closing steps (e.g. 20)
        # Total length >= 150
        self.assertGreater(len(trajectory), 150)

        # Check the shape of each step (9 joints)
        self.assertEqual(len(trajectory[0]), 9)

        # Check for gripper closing logic
        # Find the index where gripper closes (should transition from open to closed)
        # We assume open is > 0.01 and closed is < 0.01
        gripper_start_idx = -1
        gripper_end_idx = -1

        # Scan trajectory to find when gripper closes
        # Gripper joints are usually last 2
        for i, q in enumerate(trajectory):
            gripper_val = q[-1]
            if gripper_val < 0.01:
                if gripper_start_idx == -1:
                    gripper_start_idx = i
            elif gripper_start_idx != -1 and gripper_end_idx == -1:
                # Gripper re-opened? Shouldn't happen in this sequence
                pass

        # Verify gripper eventually closed
        self.assertNotEqual(gripper_start_idx, -1, "Gripper did not close")

    def test_interpolate(self):
        start = np.zeros(9)
        end = np.ones(9)
        steps = 10
        traj = self.planner.interpolate(start, end, steps)

        self.assertEqual(len(traj), steps)
        np.testing.assert_array_almost_equal(traj[0], start)
        np.testing.assert_array_almost_equal(traj[-1], end)

if __name__ == '__main__':
    unittest.main()
