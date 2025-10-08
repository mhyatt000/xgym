
import os
import os.path as osp
import time
from dataclasses import dataclass, field
from pprint import pprint

import cv2
import gymnasium as gym
import jax
import numpy as np
from pynput import keyboard
from tqdm import tqdm

# from xgym.controllers import KeyboardController, ModelController, ScriptedController
# from xgym.gyms import Base
# from xgym.utils import boundary as bd
# from xgym.utils.boundary import PartialRobotState as RS


import xgym

from pathlib import Path
import tyro
import time
from rich.pretty import pprint

import cv2
import imageio
import numpy as np
import pyudev

from xgym.utils import camera as cu
import tyro
from dataclasses import dataclass




@dataclass
class Config:

    path: Path 

def main(cfg: Config):

    for f in tqdm(list(cfg.path.rglob("*.dat"))):
        _info, ep = xgym.viz.memmap.read(f)

        imgs = {k:v for k,v in ep.items() if 'cam' in k}

        for i in range(len(imgs[list(imgs.keys())[0]])):
            _imgs = {k:v[i] for k,v in imgs.items()}

            _imgs = cu.writekeys(_imgs)
            frame = np.concatenate(list(_imgs.values()), axis=0)

            pprint((frame.shape, frame.dtype))

            cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


        pprint(ep.keys())
        # quit()

if __name__ == "__main__":
    main(tyro.cli(Config))


