from __future__ import annotations

from dataclasses import dataclass
import os
import os.path as osp
import time

import numpy as np
from pynput import keyboard
import tensorflow_datasets as tfds
from tqdm import tqdm

from xgym.controllers import KeyboardController
from xgym.gyms import Base


@dataclass
class RunCFG:
    base_dir: str = osp.expanduser("~/data")
    env_name: str = "luc-base"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():
    # env: Base = gym.make("luc-base")
    controller = KeyboardController()
    controller.register(keyboard.Key.space, lambda: env.stop(toggle=True))
    _env = Base()

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
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
        reward_info=np.float64,
        discount_info=np.float64,
    )

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
    env = _env

    for _ in tqdm(range(3)):  # 3 episodes
        obs = env.reset()
        print("entering loop")
        for i in tqdm(range(5), leave=False):
            time.sleep(0.01)

            # action = controller(obs)
            mode = "all"
            if mode == "cart":
                action = np.random.normal(0, 1, 7) * 25
                action[3:6] = 0
            elif mode == "rot":
                action = np.random.normal(0, 1, 7) * 0.05
                action[:3] = 0
            else:
                action = np.random.normal(0, 1, 7) * 25
                action[3:6] = np.random.normal(0, 1, 3) * 0.05  # rotation

            # gripper is x100 but only active 20% of the time
            gripper = np.random.choice([0, 1], p=[0.6, 0.4])
            action[6] *= gripper * 4
            action = action.astype(np.float32)

            """
            action = controller(obs)
            while action.sum() == 0:
                action = controller(obs)
            """

            print()
            print(action)

            # things = env.step()
            # obs, reward, truncated, terminated, info = env.step(env.action_space.sample())
            things = env.step(action)
            # print(things)
            _env.render()
            print("one step")

    env.close()
    controller.close()
    quit()


def sandbox():
    # env.go(RS(cartesian=[75, -125, 125], aa=[0, 0, 0]), relative=True)
    # [3.133083, 0.013429, -1.608588]

    # rpy=[np.pi, 0, -np.pi/2]
    # print(env.angles)
    # quit()
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # print(env.rpy)
    # quit()
    # rpy = np.array(env.rpy) + np.array([0.0,0,-0.1])
    # rpy[1] = 0
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # action = [0.1, 0.1, 0.1, 0, 0, 0, 0]

    pass


if __name__ == "__main__":
    main()
