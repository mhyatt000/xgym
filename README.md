# XGym

XGym is a robotics and reinforcement learning toolkit that wraps hardware control, data logging, and dataset preparation for the **XArm7** robot. It relies on ROS for communication with hardware and provides conversion tools to package collected data into `LeRobot` datasets.

## Features

- **Hardware drivers** for the XArm7 and peripheral devices like the SpaceMouse and foot pedals.
- **Gymnasium environments** for tasks such as lifting or stacking objects.
- **Data collection nodes** that record synchronized sensor streams to memmap files.
- **Dataset builders** to convert episodes into TensorFlow datasets.

## Installation

we use [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra ctrl --extra data
```

* `ctrl` extra installs joystick input libraries and the XArm SDK
* `data` extra installs TensorFlow and dataset dependencies.

Ensure your user is in the `plugdev` group for access to USB devices:

```bash
sudo usermod -aG plugdev $USER
```

## Development

Install the pre-commit hooks to automatically format code and run basic checks:

```bash
uv sync --extra dev # gives pre-commit
uv pre-commit install
```

Run all hooks manually with:

```bash
pre-commit run --all-files
```

## Directory structure

- `xgym/gyms/` – Gymnasium environments and wrappers.
- `xgym/nodes/` – ROS nodes for cameras, robot control, and controllers.
- `xgym/rlds/` – Dataset builders for TFDS.
- `scripts/` – Launch files and utilities for data collection and evaluation.

# Usage

1. **Run a camera or controller node** to start streaming data. Example:
   ```bash
   python scripts/camera.launch.py
   ```
2. **Collect data** with the writer node:
   ```bash
   python scripts/lift.py --help
   ```
3. **Convert memmaps** to LeRobot datasets using the scripts in `lrbt`:
   ```bash
   python xgym/lrbt/from_memmap.py --help
   ```

## Scripts

The `scripts/` folder contains helper utilities and launch files for data
collection and debugging. Each script supports `--help` to list available
arguments.

| Script | Purpose |
| ------ | ------- |
| `camera.launch.py` | Launch ROS2 camera nodes with a viewer. |
| `camera.py` | Display connected cameras to verify streaming. |
| `camera4human.py` | Record frames from multiple cameras for manual annotation. |
| `debug.launch.py` | Start robot and SpaceMouse nodes for development. |
| `demo.launch.py` | Example pipeline running camera, robot, and writer nodes. |
| `embed.py` | Create sentence embeddings for task descriptions. |
| `eval.py` | Evaluate a trained policy in the lift environment. |
| `filter.py` | Filter RLDS datasets and optionally drop failed episodes. |
| `get_calibration.py` | Capture robot poses for camera calibration. |
| `hamer_client.py` | Client for the HaMer hand‑tracking server. |
| `hand_tele.py` | Teleoperate the robot using tracked hand poses. |
| `human.py` | Run a human demonstration environment. |
| `lift.py` | Collect lifting task episodes. |
| `main.py` | Entry point for custom experiments. |
| `mano_read_npz.py` | Visualize MANO keypoint `.npz` files. |
| `mano_pipe_v3.py` | Full MANO hand pose estimation pipeline. |
| `poptree.py` | Move `.npz` files up one directory level. |
| `read_npz.py` | Inspect `.npz` episodes and keep or discard them. |
| `reader_rlds.py` | Render RLDS datasets with overlayed keypoints. |
| `run_from_data.py` | Replay a dataset through an environment. |
| `sandbox.py` | Miscellaneous sandbox for development. |
| `stack.py` | Collect stacking task episodes. |
| `stack_model.py` | Run a model policy in the stacking task. |
| `test_safety_box.py` | Example usage of safety‑box utilities. |
| `view_calibration.py` | Visualize saved camera calibration results. |

## Lerobot Dataset

see [lrbt/README.md](xgym/lrbt/README.md) for details on the dataset format and usage.

# Contributing

Pull requests are welcome. Please run `pre-commit` before submitting to ensure
formatting and lint checks pass.

## License

This project is licensed under the MIT License.
