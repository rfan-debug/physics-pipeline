import h5py
import numpy as np
import os

class HDF5Recorder:
    def __init__(self, save_path):
        """
        Initialize the HDF5Recorder.

        Args:
            save_path (str): Path to the HDF5 file.
        """
        self.save_path = save_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Open in append mode, creating if doesn't exist
        self.file = h5py.File(save_path, 'a')

    def create_episode_group(self, episode_idx):
        """
        Create a group for the current episode.
        Initializes resizeable datasets.

        Args:
            episode_idx (int): Index of the episode.
        """
        group_name = f"episode_{episode_idx}"
        if group_name in self.file:
            del self.file[group_name]

        grp = self.file.create_group(group_name)

        # Initialize empty datasets with resizable first dimension

        # Observations: (N, H, W, C) - uint8, compressed
        # Use maxshape=(None, None, None, None) to adapt to any resolution on first write
        grp.create_dataset('observation', shape=(0, 0, 0, 0), maxshape=(None, None, None, None), dtype='uint8', compression="gzip", chunks=True)

        # Actions: (N, D) - float32
        # Start with (0, 0) and maxshape=(None, None) to adapt to D (e.g. 7 or 9)
        grp.create_dataset('action', shape=(0, 0), maxshape=(None, None), dtype='float32', chunks=True)

        # Rewards: (N,) - float32
        grp.create_dataset('reward', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)

        # Instructions: (N,) - variable length strings
        dt = h5py.string_dtype(encoding='utf-8')
        grp.create_dataset('instruction', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)

        # State: (N, D_state) - float32 (optional)
        grp.create_dataset('state', shape=(0, 0), maxshape=(None, None), dtype='float32', chunks=True)

    def save_step(self, episode_idx, observation, action, instruction, reward, state=None):
        """
        Save a single step of data.

        Args:
            episode_idx (int): Episode index.
            observation (np.ndarray): Image.
            action (np.ndarray): Joint positions/velocities.
            instruction (str): Task instruction.
            reward (float): Reward.
            state (np.ndarray, optional): End-effector pose or other state.
        """
        group_name = f"episode_{episode_idx}"
        if group_name not in self.file:
            self.create_episode_group(episode_idx)

        grp = self.file[group_name]

        # Helper to append
        def append(name, data):
            dset = grp[name]
            current_shape = list(dset.shape)

            # Check if dataset dimensions (other than N) are uninitialized (size 0)
            is_uninitialized = False
            if len(current_shape) > 1:
                 if any(d == 0 for d in current_shape[1:]):
                      is_uninitialized = True

            if is_uninitialized:
                # First write to uninitialized dataset
                data_arr = np.array(data)
                # target shape = (1, *data_shape)
                target_shape = (1, *data_arr.shape)
                dset.resize(target_shape)
                dset[0] = data_arr
                return

            # Normal append
            new_shape = list(current_shape)
            new_shape[0] += 1
            dset.resize(tuple(new_shape))
            dset[-1] = data

        append('observation', observation)
        append('action', action)
        append('reward', reward)
        append('instruction', instruction)

        if state is not None:
            append('state', state)
        else:
            # If state not provided, we append zeros if the dataset has established dimensions,
            dset = grp['state']
            if dset.shape[0] < grp['observation'].shape[0]:
                if dset.shape[1] > 0:
                    # Append zeros matching dimension
                    new_shape = list(dset.shape)
                    new_shape[0] += 1
                    dset.resize(tuple(new_shape))
                    dset[-1] = np.zeros(dset.shape[1])
