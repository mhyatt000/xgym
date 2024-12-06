import os
import os.path as osp
from argparse import ArgumentParser as AP
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import cv2
import hamer
import imageio
import jax
import jax.numpy as jnp
import local
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm
from xgym.rlds.util import (add_col, apply_persp, apply_uv, apply_xyz,
                            remove_col, solve_uv2xyz)
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.transform import (center_crop, random_resized_crop,
                                      random_rot, random_xflip)


def perspective_projection(focal_length, H, W):
    """computes perspective projection of 3D points"""
    f = focal_length
    P = np.array(
        [
            [f, 0, W / 2, 0],
            [0, f, H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    return P


HAMER = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def unnormalize(img):
    img = img * np.array(HAMER["std"]) + np.array(HAMER["mean"])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


from hamer.datasets.utils import fliplr_keypoints, fliplr_params


def pipe(img, out):
    f = out["scaled_focal_length"]
    P = perspective_projection(f, H=img.shape[0], W=img.shape[1])

    points = out["keypoints_3d"]

    size = None
    transforms = OrderedDict(
        center_crop={
            "func": center_crop,
            "kwargs": {"size": (size := 224)},
            "rules": {"dsize": (size, size)},
        },
    )

    train_transforms = OrderedDict(
        random_resized_crop={
            "func": random_resized_crop,
            "kwargs": {
                "scale": [0.8, 1.2],
                "ratio": [0.8, 1.2],
                "tx": [-0.1, 0.1],
                "ty": [-0.1, 0.1],
            },
            "rules": {"dsize": (size, size)},
        },
        random_rot={
            "func": random_rot,
            "kwargs": {"deg": [-22.5, 22.5]},
            "rules": {"dsize": (size, size)},
        },
        random_xflip={
            "func": random_xflip,
            "kwargs": {"prob": 0.5},
            "rules": {"dsize": (size, size)},
        },
        # random_occlusion={
        # "func": random_occlusion,  # doesnt have to occlude the hand it can be anywhere
        # "kwargs": {
        # "prob": 0.5,
        # "area": [0.0, 0.1],
        # "ratio": [0.5, 2.0],
        # "nboxes": [1, 3],
        # },
        # "rules": {"dsize": (size, size)},
        # },
    )

    prng = jax.random.PRNGKey(0)

    U = np.eye(4)
    for t, v in transforms.items():
        prng, seed = jax.random.split(prng)
        _U = v["func"](**v["kwargs"], seed=seed, img=img)

        img = apply_uv(img, mat=_U, **v["rules"])
        U = _U @ U

    T = solve_uv2xyz(points, P=P, U=U)

    points3d = apply_xyz(points, mat=T)
    points2d = apply_persp(points3d, P)

    ### for debugging only
    # img = local.render_openpose(img, points2d[0])
    # img = local.render_openpose(img, points2d[1])

    # cv2.imshow("frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(100)
    # quit()

    out["keypoints_3d"] = points3d
    out["keypoints_2d"] = points2d
    out["frame"] = img

    # crops from hamer
    out["frame_hand"] = np.stack(
        [unnormalize(x.transpose(1, 2, 0)) for x in out["img"]]
    )
    return out


spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)


def clean(out):

    obskeys = [
        "focal_length",
        "scaled_focal_length",
        #
        "frame",
        "keypoints_2d",
        "keypoints_3d",
        # "mano",
        "mano.betas",
        "mano.global_orient",
        "mano.hand_pose",
        "right",
    ]

    out = {k.replace("pred_", "").replace("_params", ""): v for k, v in out.items()}

    out = jax.tree.map(
        lambda x: (
            x[0].astype(np.float32) if len(x.shape) and x.shape[0] in [1, 2] else x
        ),
        out,
    )

    # pprint(spec(out))
    # out["right"] = out['mano'].pop("right")
    for k in ["focal_length", "scaled_focal_length", "right"]:
        if isinstance(out[k], np.ndarray):
            out[k] = out[k].flatten()[0]

    out = {k: v for k, v in out.items() if k in obskeys}

    out["keypoints_2d"] = out["keypoints_2d"].reshape(21, 3)[:, :-1]

    if not all([k in out for k in obskeys]):
        print("missing keys")
        print(set(obskeys) - set(out.keys()))
        return None

    return out


def main():

    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir.parent / f"{data_dir.name}_out"
    out_dir.mkdir(exist_ok=True)

    files = [
        int(x.name.split(".")[0].split("_")[1]) for x in data_dir.glob("pose_*.npz")
    ]
    total = max(files)

    video = imageio.get_reader(args.data_dir / "output.mp4")
    for i, frame in tqdm(enumerate(video), total=total):

        try:
            name = f"pose_{i}.npz"
            data = np.load(data_dir / name)
            data = {k: data[k] for k in data.files}

            out = pipe(frame, data)
        except Exception as e:
            print(e)
            continue

        out = clean(out)

        if out is None:
            continue
        print("spec")
        pprint(spec(out))

        # print(f"saving to {out_dir/name}")
        # if input("continue? [y/n] ") == "n":
        # quit()

        np.savez(out_dir / name, **out)


def preclean(step):
    dropkeys = [
        "box_center",
        "box_size",
        "img_size",
        "personid",
        "cam",
        "cam_t",
        "vertices",
    ]

    step = {k.replace("pred_", ""): v for k, v in step.items()}

    # print(spec(step))

    # right = step["keypoints_3d"]
    # left = fliplr_keypoints(step["keypoints_3d"], width=step["box_size"][0])
    # step["keypoints_3d"] = np.where(step["right"], right, left)
    step["keypoints_3d"] += step.pop("cam_t_full")[:, None]

    for k in dropkeys:
        step.pop(k)

    return step


spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)


def visualize(sequence):

    for i, _ in enumerate(sequence["focal_length"]):
        step = jax.tree_map(lambda x: x[i], sequence)

        img = step["frame"]

        img = render_openpose(img, step["keypoints_2d"])

        cv2.imshow("frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)


def process_sequence(sequence):

    is_right = int(np.array([step["right"].mean() for step in sequence]).mean() > 0.5)

    # print(f"right: {is_right}")
    if not is_right:
        return

    def select_hand(x):
        if x.shape and x.shape[0] == 2:
            return x[is_right]
        if x.shape and x.shape[0] == 1:
            return x[0]
        return x

    seq2 = []
    for i, step in enumerate(sequence):
        step = jax.tree.map(select_hand, step)
        # pprint(spec(step))
        step["focal_length"] = step["focal_length"].flatten()[0]
        seq2.append(step)

    out = {}
    for k in seq2[0].keys():
        out[k] = np.stack([step[k] for step in seq2])

    return out


def main():

    BALA_DATA = Path("~/.bala/output_hamer").expanduser()
    data_dir = BALA_DATA

    # episodes are folders inside the data_dir ... TODO name them episode_i
    episodes = sorted([x for x in data_dir.glob("*") if x.is_dir()])

    for ep in tqdm(episodes):
        steps = sorted(
            [x for x in ep.glob("step_*.npz")],
            key=lambda x: int(x.name.split(".")[0].split("_")[1]),
        )

        sequence = []
        for s in tqdm(steps, total=len(steps), leave=False):
            step = np.load(s)
            step = {k: step[k] for k in step.files}

            step = preclean(step)

            step = pipe(step["frame"], step)
            sequence.append(step)

        sequence = process_sequence(sequence)
        if sequence is not None:
            # visualize(sequence)
            
            # save to npz
            # prepare for tfds
            np.savez(ep.parent / f"{ep.name}_tfds.npz", **sequence)
        continue

if __name__ == "__main__":
    main()
