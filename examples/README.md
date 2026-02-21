# VLA Synthesis Examples

This directory contains example scripts demonstrating how to use the VLA Synthesis pipeline components.

## Prerequisites

Ensure you have the `genesis` library installed. If not, the examples will print an error message but will not run fully.

You also need `numpy` and `h5py` installed:
```bash
pip install numpy h5py
```

## Running Examples

Run the examples from the root of the repository or from the `examples/` directory.

### 1. Basic Scene Loading (`01_basic_scene.py`)
Loads the robot and scene, sets up the camera, and renders a single frame.
```bash
python examples/01_basic_scene.py
```

### 2. Task Generation (`02_task_generation.py`)
Demonstrates how to generate random tasks (instructions and target objects) and reset the scene.
```bash
python examples/02_task_generation.py
```

### 3. Full Episode Simulation (`03_full_episode.py`)
Runs a full episode simulation including planning, execution, and recording to an HDF5 file.
```bash
python examples/03_full_episode.py
```
