from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro

# from xgym.controllers import KeyboardController, ModelController, ScriptedController
# from xgym.gyms import Base
# from xgym.utils import boundary as bd
# from xgym.utils.boundary import PartialRobotState as RS
import xgym
from xgym.utils import camera as cu


@dataclass
class Config:
    path: Path


def main(cfg: Config):
    for f in tqdm(list(cfg.path.rglob("*.dat"))):
        _info, ep = xgym.viz.memmap.read(f)

        imgs = {k: v for k, v in ep.items() if "cam" in k}

        # use
        for i in range(len(imgs[next(iter(imgs.keys()))])):
            _imgs = {k: v[i] for k, v in imgs.items()}

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
