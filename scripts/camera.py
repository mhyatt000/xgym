from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
from rich.pretty import pprint
import tyro

from xgym.utils import camera as cu

"""
context = pyudev.Context()

import pandas as pd

df = []
for device in context.list_devices(subsystem="video4linux"):
    df.append(
        {
            "path": device.device_node,
            "serial": device.get("ID_SERIAL_SHORT"),
            "model": device.get("ID_MODEL"),
        }
    )

df = pd.DataFrame(df)
# sort by int(path.split("/")[-1].replace("video", ""))
df = df.sort_values(
    by="path",
    key=lambda x: x.str.split("/").str[-1].str.replace("video", "").astype(int),
)

print(df)
"""


@dataclass
class Config:
    cams: list[int] | None = None


def main(cfg: Config):
    store = False
    cams = cu.list_cameras(cfg.cams)

    print(cams)

    for k, cam in cams.items():
        print(f"Camera {k}: {cam.get(cv2.CAP_PROP_FPS)} FPS")

    fps = 30
    dt = 1 / fps

    # pop all but one cam

    print(cams)
    frames = []
    while True:
        tic = time.time()

        _ = [cam.grab() for cam in cams.values()]
        imgs = {k: cam.read() for k, cam in cams.items()}
        imgs = {k: im for k, (ret, im) in imgs.items() if ret}
        pprint({k: v.shape for k, v in imgs.items()})
        # imgs = {k: cu.square(f) for k, f in imgs.items()}
        imgs = cu.writekeys(imgs)

        # resize by the biggest dimension of frames
        imgs = cu.resize_all(list(imgs.values()), m=480)
        frame = np.concatenate(imgs, axis=0)

        if store:
            frames.append(frame)

        cv2.putText(
            frame,
            f"FPS: 5 | time: {tic}",
            (512, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        if len(frames) % 5 == 0:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        toc = time.time()
        elapsed = toc - tic
        time.sleep(max(0, dt - elapsed))

    if store:
        cu.save_frames(frames, "test", ext="mp4", fps=30)

    quit()


# frames = np.array(frames)


def realsense():
    from gello.cameras.realsense_camera import get_device_ids, RealSenseCamera

    device_ids = get_device_ids()
    rs = RealSenseCamera(flip=False, device_id=device_ids[0])

    while True:
        image, depth = rs.read()
        image = cv2.resize(image, (480, 480))
        print(image.shape)
        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imshow("depth", depth)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    # realsense()
    main(tyro.cli(Config))
