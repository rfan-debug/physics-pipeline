import numpy as np
import warnings

class SimplePlanner:
    def __init__(self):
        """
        Initialize the SimplePlanner.
        """
        pass

    def interpolate(self, start_q, end_q, steps):
        """
        Linearly interpolate between two joint configurations.

        Args:
            start_q (np.ndarray): Starting joint configuration.
            end_q (np.ndarray): Ending joint configuration.
            steps (int): Number of steps in the interpolation.

        Returns:
            list: List of joint configurations.
        """
        start_q = np.array(start_q)
        end_q = np.array(end_q)
        trajectory = []
        for i in range(steps):
            # Linear interpolation: q(t) = (1-t)*start + t*end
            t = i / (steps - 1) if steps > 1 else 0
            q = (1 - t) * start_q + t * end_q
            trajectory.append(q)
        return trajectory

    def plan_grasp(self, robot, target_position):
        """
        Generate a trajectory for grasping an object at target_position.

        Args:
            robot: The robot entity (Genesis morph).
            target_position (list or np.ndarray): Target position [x, y, z].

        Returns:
            list: A list of joint configurations (actions) for the entire episode.
        """
        # Ensure target_position is a numpy array
        target_pos = np.array(target_position)

        # 1. Define Key Poses
        # Pre-grasp: 10cm above object
        pre_grasp_pos = target_pos + np.array([0, 0, 0.1])

        # Grasp: At object
        grasp_pos = target_pos

        # Lift: 10cm above object (same as pre-grasp)
        lift_pos = pre_grasp_pos

        # Orientation: Gripper pointing down (e.g., quaternion [0, 1, 0, 0] or similar)
        # We assume standard orientation where gripper z-axis points down.
        # This might need adjustment based on robot's coordinate frame.
        # For Franka Panda, usually pointing down is a specific quaternion.
        # Let's use a placeholder [0, 1, 0, 0] which is often used for "down" in many frames,
        # or [1, 0, 0, 0] (identity). If Identity is "up", then we need a rotation.
        # A common "down" orientation for Panda is rotation around Y or X.
        # Given we lack docs, we'll try a standard one.
        # Using [0, 1, 0, 0] (w, x, y, z) -> 180 deg around X?
        # Let's use a dummy quaternion and rely on IK solver to handle it or fail gracefully.
        down_quat = np.array([0, 1, 0, 0])

        # 2. Calculate Joint Angles using IK
        try:
            # Current configuration
            if hasattr(robot, 'get_q'):
                home_q = np.array(robot.get_q())
            else:
                # Fallback if get_q is missing (e.g. simulation not started)
                # Assume 9 DOFs (7 arm + 2 gripper)
                home_q = np.zeros(9)
                warnings.warn("robot.get_q() not found. Using zero configuration.")

            # Helper to call IK safely
            def solve_ik(pos, quat):
                if hasattr(robot, 'inverse_kinematics'):
                    # Assume signature: link (optional), pos, quat
                    # We pass link=None to imply end-effector
                    # Check signature via inspection if possible, but here we just try
                    q = robot.inverse_kinematics(link=None, pos=pos, quat=quat)
                    return np.array(q)
                else:
                    # Mock behavior for testing if method missing
                    warnings.warn("robot.inverse_kinematics not found. Returning random configuration.")
                    return np.random.uniform(-1, 1, size=len(home_q))

            pre_grasp_q = solve_ik(pre_grasp_pos, down_quat)
            grasp_q = solve_ik(grasp_pos, down_quat)
            lift_q = solve_ik(lift_pos, down_quat)

            # Ensure gripper is open for these keyframes (last 2 joints)
            # Assuming last 2 are gripper fingers and open is e.g. 0.04
            open_val = 0.04
            closed_val = 0.0

            # Helper to set gripper
            def set_gripper(q, val):
                q_copy = q.copy()
                if len(q_copy) >= 2:
                    q_copy[-2:] = val
                return q_copy

            pre_grasp_q = set_gripper(pre_grasp_q, open_val)
            grasp_q = set_gripper(grasp_q, open_val)
            lift_q = set_gripper(lift_q, closed_val)

        except Exception as e:
            raise RuntimeError(f"IK solving failed: {e}")

        # 3. Interpolate Trajectories
        steps_per_segment = 50

        # Home -> Pre-grasp
        traj_home_to_pre = self.interpolate(home_q, pre_grasp_q, steps_per_segment)

        # Pre-grasp -> Grasp
        traj_pre_to_grasp = self.interpolate(pre_grasp_q, grasp_q, steps_per_segment)

        # Close Gripper
        # We simulate closing by interpolating from open to closed configuration at the same pose
        grasp_q_closed = set_gripper(grasp_q, closed_val)
        traj_close_gripper = self.interpolate(grasp_q, grasp_q_closed, 20)

        # Grasp -> Lift
        traj_grasp_to_lift = self.interpolate(grasp_q_closed, lift_q, steps_per_segment)

        # Combine all
        full_trajectory = (
            traj_home_to_pre +
            traj_pre_to_grasp +
            traj_close_gripper +
            traj_grasp_to_lift
        )

        return full_trajectory
