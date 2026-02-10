import unittest
import numpy as np
import h5py
import os
import shutil
import tempfile
import sys

# Adjust path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vla_synthesis.src.recorder import HDF5Recorder

class TestHDF5Recorder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.test_dir, 'test_data.h5')
        self.recorder = HDF5Recorder(self.h5_path)

    def tearDown(self):
        # Close the recorder if open (it might close file handle)
        # Assuming explicit close isn't strictly enforced by context manager,
        # but removing file is enough for temp usage.
        # Ideally close the file first.
        if hasattr(self.recorder, 'file') and self.recorder.file:
             self.recorder.file.close()
        shutil.rmtree(self.test_dir)

    def test_init_creates_file(self):
        self.assertTrue(os.path.exists(self.h5_path))
        with h5py.File(self.h5_path, 'r') as f:
            self.assertTrue(isinstance(f, h5py.File))

    def test_create_episode_group(self):
        episode_idx = 0
        self.recorder.create_episode_group(episode_idx)

        with h5py.File(self.h5_path, 'r') as f:
            group_name = f'episode_{episode_idx}'
            self.assertIn(group_name, f)
            # Check datasets exist but are empty
            grp = f[group_name]
            self.assertIn('observation', grp)
            self.assertIn('action', grp)
            self.assertIn('reward', grp)
            self.assertIn('instruction', grp)
            self.assertIn('state', grp)

            # Check empty shape
            self.assertEqual(grp['observation'].shape[0], 0)

    def test_save_step(self):
        episode_idx = 1
        self.recorder.create_episode_group(episode_idx)

        # Create dummy data
        obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        action = np.random.rand(7).astype(np.float32)
        instruction = "Pick up the red cube"
        reward = 1.0
        state = np.random.rand(6).astype(np.float32) # optional

        self.recorder.save_step(episode_idx, obs, action, instruction, reward, state)

        # Verify data
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[f'episode_{episode_idx}']

            # Check shapes
            self.assertEqual(grp['observation'].shape, (1, 480, 640, 3))
            self.assertEqual(grp['action'].shape, (1, 7))
            self.assertEqual(grp['reward'].shape, (1,))
            self.assertEqual(grp['instruction'].shape, (1,))
            self.assertEqual(grp['state'].shape, (1, 6))

            # Check values
            np.testing.assert_array_equal(grp['observation'][0], obs)
            np.testing.assert_array_almost_equal(grp['action'][0], action)
            self.assertEqual(grp['reward'][0], reward)

            # Check string decoding
            saved_instr = grp['instruction'][0]
            if isinstance(saved_instr, bytes):
                saved_instr = saved_instr.decode('utf-8')
            self.assertEqual(saved_instr, instruction)

    def test_save_multiple_steps(self):
        episode_idx = 2
        self.recorder.create_episode_group(episode_idx)

        steps = 5
        for i in range(steps):
            obs = np.full((10, 10, 3), i, dtype=np.uint8)
            action = np.full((3,), i * 0.1, dtype=np.float32)
            instr = f"Step {i}"
            rew = float(i)
            self.recorder.save_step(episode_idx, obs, action, instr, rew)

        with h5py.File(self.h5_path, 'r') as f:
            grp = f[f'episode_{episode_idx}']
            self.assertEqual(grp['observation'].shape[0], steps)
            self.assertEqual(grp['observation'][-1, 0, 0, 0], steps - 1)

if __name__ == '__main__':
    unittest.main()
