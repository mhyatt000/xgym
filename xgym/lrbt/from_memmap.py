import enum
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import draccus
import jax
import jax.numpy as jnp
import lerobot
import numpy as np
import torch
import tyro
from flax.traverse_util import flatten_dict
from jax import numpy as jnp
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rich.pretty import pprint
from tqdm import tqdm

import xgym
from xgym.lrbt.convert import (
    DEFAULT_DATASET_CONFIG,
    FRAME_MODE,
    MOTORS,
    DatasetConfig,
    Embodiment,
    Task,
    create,
)
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose
from xgym.lrbt.builder import BaseConverter
from xgym.lrbt.util import get_taskinfo


np.set_printoptions(suppress=True, precision=3)


@dataclass
class Config:
    dir: str  # the dir of the memmap files
    branch: str  # the branch to push to ie: "v0.1"
    repo_id: str  # the repo id to push to ie: "mhyatt000/demo"


class MemmapConverter(BaseConverter):
    def __init__(self, cfg: Config) -> None:
        super().__init__(repo_id=cfg.repo_id, robot_type="xarm")
        self.cfg = cfg

    def run(self) -> None:
        root = Path(self.cfg.dir)
        files = list(root.rglob("*.dat"))

        spec = lambda tree: jax.tree.map(lambda x: x.shape, tree)
        take = lambda tree, i: jax.tree.map(lambda x: x[i], tree)

        taskinfo = get_taskinfo(self.cfg.dir)
        _task, _lang = taskinfo["task"], taskinfo["lang"]

        for f in tqdm(files, total=len(files)):
            try:
                info, ep = xgym.viz.memmap.read(f)
                cams = [k for k in info["schema"].keys() if "camera" in k]
                if len(cams) < 2:
                    raise ValueError(f"Not enough cameras {cams}")
            except Exception as e:
                xgym.logger.error(f"Error reading {f}")
                xgym.logger.error(e)
                continue

            if len(ep["time"]) < 200:
                xgym.logger.error(f"Episode too short {len(ep['time'])}")
                continue
            pprint(spec(ep))

            leader = ep["gello_joints"]
            leader_grip = leader[:, -1:]

            n = len(ep["time"])
            steps = {
                "observation": {
                    "image": {k.split("/")[-1]: ep[k] for k in ep if "camera" in k},
                    "proprio": {k.split("_")[-1]: ep[k] for k in ep if "xarm" in k},
                },
            }
            steps["observation"]["proprio"]["gripper"] = leader_grip
            pprint(spec(steps))

            for k in [
                "discount",
                "reward",
                "is_first",
                "is_last",
                "is_terminal",
                "action",
            ]:
                if k in steps:
                    steps.pop(k)

            obs = steps.pop("observation")
            img = obs.pop("image")

            for k in ["overhead", "high"]:
                if k in img:
                    img.pop(k)

            img = jax.tree.map(lambda x: x / 255, img)
            obs["image"] = img

            state = obs.pop("proprio")
            state["position"] = state.pop("pose") / 1e3
            obs["state"] = state

            steps["observation"] = obs

            lang = {"lang": np.array([_lang for _ in range(n)]), "task": _task}
            steps["lang"] = lang["lang"]

            mask = ~np.isnan(state["gripper"]).reshape(-1)
            steps = jax.tree.map(lambda x: x[mask], steps)
            n = mask.sum()

            print(f"Kept {n} of {len(mask)} steps")

            step0 = take(steps, 0)
            self.ensure_dataset(step0)

            self.add_episode(steps, str(lang["task"]))

        self.push(branch=self.cfg.branch)


def main(cfg: Config) -> None:
    MemmapConverter(cfg).run()


if __name__ == "__main__":
    main(tyro.cli(Config))
