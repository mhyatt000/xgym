from __future__ import annotations

from dataclasses import dataclass
import os
import os.path as osp

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from xgym.controllers import ModelController
from xgym.gyms import Stack


@dataclass
class RunCFG:
    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-stack-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():
    os.makedirs(cfg.data_dir, exist_ok=True)
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="luc-base",
        observation_info=tfds.features.FeaturesDict(
            {
                "robot": tfds.features.FeaturesDict(
                    {
                        "joints": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                        "position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                    }
                ),
                "img": tfds.features.FeaturesDict(
                    {
                        "camera_0": tfds.features.Tensor(shape=(640, 640, 3), dtype=np.uint8),
                        "wrist": tfds.features.Tensor(shape=(640, 640, 3), dtype=np.uint8),
                    }
                ),
            }
        ),
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float64),
        reward_info=tfds.features.Tensor(shape=(), dtype=np.float64),
        discount_info=tfds.features.Tensor(shape=(), dtype=np.float64),
    )

    # env: Base = gym.make("luc-base")

    model = ModelController("carina.cs.luc.edu", 8001)
    model.reset()

    # env = gym.make("xgym/stack-v0")
    _env = Stack()

    """
    with envlogger.EnvLogger(
        DMEnvFromGym(_env),
        backend=TFDSWriter(
            data_directory=cfg.data_dir,
            split_name="train",
            max_episodes_per_file=50,
            ds_config=dataset_config,
        ),
    ) as env:
    """

    with _env as env:
        for ep in range(10):
            obs = env.reset()
            for _ in tqdm(range(50)):  # 3 episodes
                print("\n" * 3)
                action = model(obs["img"]["camera_0"]).copy()
                action[3:6] = 0
                action[-1] *= 850
                # action = action / 2
                print("action")
                print(action.tolist())
                env.render(mode="human")
                obs, _reward, _done, _info = env.step(action)

    _env.close()

    quit()


if __name__ == "__main__":
    main()
