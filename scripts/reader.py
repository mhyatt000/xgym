from __future__ import annotations

from dataclasses import dataclass
import os.path as osp
import sys

import cv2
import jax
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


@dataclass
class RunCFG:
    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-lift-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


print(cfg.data_dir)
ds = tfds.builder_from_directory(sys.argv[1]).as_dataset(split="all")


print(ds)

A = []
N = 0


for e in tqdm(ds):
    # print(e)

    n = len(e["steps"])
    N += n
    actions = np.zeros(7)

    for s in e["steps"]:
        s = jax.tree_map(lambda x: np.array(x), s)
        # pprint(jax.tree_map(lambda x: (x.shape,x.dtype), s))

        obs = s["observation"]
        imgs = obs["img"]
        imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        cv2.waitKey(100)

        action = s["action"]
        print([round(a, 4) for a in action])
        actions += action
        A.append(action)
        # pprint(obs['robot'])
        # print()

    a = [round(x, 4) for x in (actions / n).tolist()]
    # print(a)
    # input("Press Enter to continue...")

# find mean and std of actions
mean = np.mean(A, axis=0)
std = np.std(A, axis=0)

print(mean)
print(std)
