import h5py
import numpy as np
import os

class HDF5Recorder:
    def __init__(self, save_path):
        """
        Initialize the HDF5 recorder.

        Args:
            save_path (str): Path to the HDF5 file.
        """
        self.save_path = save_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self.file = h5py.File(save_path, 'a')

    def create_episode_group(self, episode_idx):
        """
        Create a group for the episode.

        Args:
            episode_idx (int): Index of the episode.
        """
        group_name = f'episode_{episode_idx}'
        if group_name in self.file:
            del self.file[group_name]
        self.file.create_group(group_name)

    def save_step(self, episode_idx, observation, action, instruction, reward, state=None):
        """
        Save a single step of data.

        Args:
            episode_idx (int): Index of the episode.
            observation (np.ndarray): RGB image (uint8).
            action (np.ndarray): Joint positions/velocities (float32).
            instruction (str): Text instruction.
            reward (float): Reward.
            state (np.ndarray, optional): Robot end-effector pose.
        """
        group_name = f'episode_{episode_idx}'
        if group_name not in self.file:
            self.create_episode_group(episode_idx)

        grp = self.file[group_name]

        # Ensure correct types
        if observation.dtype != np.uint8:
            observation = observation.astype(np.uint8)

        if action.dtype != np.float32:
            action = action.astype(np.float32)

        # Initialize datasets if they don't exist
        if 'observations' not in grp:
            # Observation: (N, H, W, 3)
            obs_shape = (0,) + observation.shape
            max_obs_shape = (None,) + observation.shape
            grp.create_dataset('observations', shape=obs_shape, maxshape=max_obs_shape, dtype='uint8', compression='gzip')

            # Action: (N, D)
            act_shape = (0,) + action.shape
            max_act_shape = (None,) + action.shape
            grp.create_dataset('actions', shape=act_shape, maxshape=max_act_shape, dtype='float32')

            # Instruction: (N,)
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset('instructions', shape=(0,), maxshape=(None,), dtype=dt)

            # Reward: (N,)
            grp.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float32')

            # State (optional): (N, D_state)
            if state is not None:
                state_shape = (0,) + state.shape
                max_state_shape = (None,) + state.shape
                grp.create_dataset('states', shape=state_shape, maxshape=max_state_shape, dtype='float32')

        # Append data by resizing
        n = grp['observations'].shape[0]

        grp['observations'].resize((n + 1,) + observation.shape)
        grp['observations'][n] = observation

        grp['actions'].resize((n + 1,) + action.shape)
        grp['actions'][n] = action

        grp['instructions'].resize((n + 1,))
        grp['instructions'][n] = instruction

        grp['rewards'].resize((n + 1,))
        grp['rewards'][n] = reward

        if state is not None:
            if 'states' not in grp:
                 # If state appears late, create the dataset now
                 state_shape = (0,) + state.shape
                 max_state_shape = (None,) + state.shape
                 grp.create_dataset('states', shape=state_shape, maxshape=max_state_shape, dtype='float32')

            grp['states'].resize((n + 1,) + state.shape)
            grp['states'][n] = state

    def close(self):
        """Close the HDF5 file."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
