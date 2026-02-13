import unittest
import numpy as np
import h5py
import os
import shutil
import tempfile
from vla_synthesis.src.recorder import HDF5Recorder

class TestHDF5Recorder(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.test_dir, 'test.h5')
        self.recorder = HDF5Recorder(self.h5_path)

    def tearDown(self):
        self.recorder.close()
        shutil.rmtree(self.test_dir)

    def test_create_episode_group(self):
        self.recorder.create_episode_group(0)
        self.recorder.close()

        with h5py.File(self.h5_path, 'r') as f:
            self.assertIn('episode_0', f)

    def test_save_step(self):
        obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        action = np.random.rand(7).astype(np.float32)
        instruction = "pick up the cube"
        reward = 1.0
        state = np.random.rand(6).astype(np.float32)

        self.recorder.save_step(0, obs, action, instruction, reward, state)
        self.recorder.close()

        with h5py.File(self.h5_path, 'r') as f:
            self.assertIn('episode_0', f)
            grp = f['episode_0']

            self.assertIn('observations', grp)
            self.assertIn('actions', grp)
            self.assertIn('instructions', grp)
            self.assertIn('rewards', grp)
            self.assertIn('states', grp)

            self.assertEqual(grp['observations'].shape, (1, 480, 640, 3))
            self.assertEqual(grp['actions'].shape, (1, 7))
            self.assertEqual(grp['instructions'].shape, (1,))
            self.assertEqual(grp['rewards'].shape, (1,))
            self.assertEqual(grp['states'].shape, (1, 6))

            np.testing.assert_array_equal(grp['observations'][0], obs)
            np.testing.assert_array_equal(grp['actions'][0], action)
            self.assertEqual(grp['instructions'][0].decode('utf-8'), instruction)
            self.assertEqual(grp['rewards'][0], reward)
            np.testing.assert_array_equal(grp['states'][0], state)

            # Check compression
            self.assertEqual(grp['observations'].compression, 'gzip')

    def test_multiple_steps(self):
        steps = 5
        for i in range(steps):
            obs = np.zeros((10, 10, 3), dtype=np.uint8)
            action = np.zeros(5, dtype=np.float32)
            instruction = "step"
            reward = float(i)
            self.recorder.save_step(0, obs, action, instruction, reward)

        self.recorder.close()

        with h5py.File(self.h5_path, 'r') as f:
            grp = f['episode_0']
            self.assertEqual(grp['observations'].shape, (5, 10, 10, 3))
            self.assertEqual(grp['rewards'].shape, (5,))
            self.assertEqual(grp['rewards'][4], 4.0)

    def test_overwrite_episode(self):
        self.recorder.create_episode_group(0)
        # Write something manually to check overwrite
        self.recorder.file['episode_0'].attrs['foo'] = 'bar'

        self.recorder.create_episode_group(0)
        self.recorder.close()

        with h5py.File(self.h5_path, 'r') as f:
            self.assertIn('episode_0', f)
            self.assertNotIn('foo', f['episode_0'].attrs)

    def test_state_late_arrival(self):
        # Step 0: no state
        obs = np.zeros((10, 10, 3), dtype=np.uint8)
        action = np.zeros(5, dtype=np.float32)
        instruction = "step"
        reward = 0.0

        self.recorder.save_step(0, obs, action, instruction, reward)

        # Step 1: with state
        state = np.ones(3, dtype=np.float32)
        self.recorder.save_step(0, obs, action, instruction, reward, state)

        self.recorder.close()

        with h5py.File(self.h5_path, 'r') as f:
            grp = f['episode_0']
            self.assertIn('states', grp)
            self.assertEqual(grp['states'].shape, (2, 3))

            # First state should be 0 (default fill)
            np.testing.assert_array_equal(grp['states'][0], np.zeros(3))
            # Second state should be 1
            np.testing.assert_array_equal(grp['states'][1], np.ones(3))

if __name__ == '__main__':
    unittest.main()
