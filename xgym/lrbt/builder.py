from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import torch
from flax.traverse_util import flatten_dict
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .convert import create


class BaseConverter:
    """Helper to build a LeRobot dataset from processed episodes."""

    def __init__(self, repo_id: str, robot_type: str) -> None:
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.dataset: LeRobotDataset | None = None

    def ensure_dataset(self, example: dict[str, Any]) -> LeRobotDataset:
        """Create the underlying dataset if necessary."""
        if self.dataset is None:
            self.dataset = create(
                repo_id=self.repo_id,
                robot_type=self.robot_type,
                example=deepcopy(example),
            )
        return self.dataset

    def add_episode(self, steps: dict[str, Any], task: str) -> None:
        """Add an episode to the dataset."""
        take = lambda tree, i: jax.tree.map(lambda x: x[i], tree)
        n = len(jax.tree_util.tree_leaves(take(steps, 0))[0])
        for i in tqdm(range(n), leave=False):
            step = take(steps, i)
            step = flatten_dict(step, sep=".")
            step = jax.tree.map(lambda x: torch.from_numpy(x).float(), step)
            assert self.dataset is not None
            self.dataset.add_frame(step | {"task": task})
        assert self.dataset is not None
        self.dataset.save_episode()

    def push(self, *, branch: str, tags: list[str] | None = None) -> None:
        """Push the dataset to the Hugging Face hub."""
        tags = [] if tags is None else tags
        assert self.dataset is not None
        self.dataset.push_to_hub(
            branch=branch,
            tags=tags,
            private=False,
            push_videos=True,
        )
