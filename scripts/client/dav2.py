from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import time

import cv2
import numpy as np
from rich.pretty import pprint
import tyro
from webpolicy.deploy.client import WebsocketClientPolicy as Client


def show(img: np.ndarray, title: str = "Image"):
    if False:
        # convert from min,max to mean0-std1
        img = (img - np.mean(img)) / np.std(img)
        # convert from min,mix to 0-1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # add color
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_COOL)

    pprint((img.shape, img.dtype, img.min(), img.max()))
    cv2.imshow("Selfie", img)

    if key := cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        quit()


@dataclass
class Config:
    host: str
    port: int = 8001

    cam: int = 0
    maxd: float = 20.0  # max depth
    relative: bool = False  # relative colors
    resize: int = 5  # downsample n times
    extreme: bool = False  # use extreme color


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {f.__name__} took {end - start:.4f} seconds | {1 / (end - start):.2f} FPS")
        return result

    return wrapper


def main(cfg: Config):
    print("Hello from dav-policy!")

    cam = cv2.VideoCapture(cfg.cam)
    client = Client(host=cfg.host, port=cfg.port)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # resize downsample n times
        n = cfg.resize
        _h, _w = frame.shape[:2]
        h, w = _h // n, _w // n
        frame = cv2.resize(frame, (w, h))

        infer = timeit(client.infer)
        out = infer({"img": frame, "max_depth": cfg.maxd, "relative": cfg.relative})
        depth, cmap = iter(out.values())
        pprint((depth.min(), depth.max(), depth.mean()))

        cmap = cv2.resize(cmap, (_w, _h))
        if cfg.extreme:
            cmap = cmap / 255
            cmap = cmap**2
            cmap = (cmap * 255).astype(np.uint8)

        show(cmap, title="Webcam Feed")


if __name__ == "__main__":
    main(tyro.cli(Config))
