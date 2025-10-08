from __future__ import annotations

from dataclasses import dataclass
import os.path as osp
import time

import imageio
from tqdm import tqdm

from xgym.gyms import Human


@dataclass
class RunCFG:
    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-stack-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():
    # env = gym.make("xgym/stack-v0")
    env = Human(mode="manual")
    env.reset()

    for ep in range(1):
        all_imgs = []

        for i in tqdm(range(int(1e3)), desc=f"Collecting episode {ep}"):
            imgs = env.look()
            all_imgs.append(imgs)
            # env.render(refresh=True)
            time.sleep(0.1)

        # make them into a video and save to disk
        for k in all_imgs[0]:
            frames = [x[k] for x in all_imgs]
            with imageio.get_writer(f"ep{ep}_{k}.mp4", fps=10) as writer:
                for frame in frames:
                    writer.append_data(frame)

    env.close()
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
