import glob
import json
import os
import os.path as osp
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def downscale_to_224(height: int, width: int) -> Tuple[int, int]:
    """
    Downscale the image so that the shorter dimension is 224 pixels,
    and the longer dimension is scaled by the same ratio.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        Tuple[int, int]: The new height and width of the image.
    """
    # Determine the scaling ratio
    if height < width:
        ratio = 224.0 / height
        new_height = 224
        new_width = int(width * ratio)
    else:
        ratio = 224.0 / width
        new_width = 224
        new_height = int(height * ratio)

    return new_height, new_width


class XgymLiftSingle(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LUC XGym Single Arm v1.0.0"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.FeaturesDict(
                                        {
                                            "camera_0": tfds.features.Image(
                                                shape=(
                                                    224,
                                                    224,
                                                    3,
                                                ),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Main camera RGB observation.",
                                            ),
                                            "wrist": tfds.features.Image(
                                                shape=(
                                                    224,
                                                    224,
                                                    3,
                                                ),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Wrist camera RGB observation.",
                                            ),
                                        }
                                    ),
                                    "proprio": tfds.features.FeaturesDict(
                                        {
                                            "joints": tfds.features.Tensor(
                                                shape=[7],
                                                dtype=np.float32,
                                                doc="Joint angles. radians",
                                            ),
                                            "position": tfds.features.Tensor(
                                                shape=[7],
                                                dtype=np.float32,
                                                doc="Joint positions. xyz millimeters (mm) and rpy",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(
                                    7,
                                ),  # do we need 8? for terminate episode action?
                                dtype=np.float32,
                                doc="Robot action, consists of [xyz,rpy,gripper].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({}),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        root = osp.expanduser("~/tensorflow_datasets/xgym_single/source")
        root = osp.expanduser("~/data/xgym-lift-v0-*")

        self.filtered = osp.expanduser("~/data/filtered.json")

        return {
            "train": self._generate_examples(root),
        }

    def is_noop(self, action, prev_action=None, threshold=1e-4):
        """
        Returns whether an action is a no-op action.

        A no-op action satisfies two criteria:
            (1) All action dimensions, except for the last one (gripper action), are near zero.
            (2) The gripper action is equal to the previous timestep's gripper action.

        Explanation of (2):
            Naively filtering out actions with just criterion (1) is not good because you will
            remove actions where the robot is staying still but opening/closing its gripper.
            So you also need to consider the current state (by checking the previous timestep's
            gripper action as a proxy) to determine whether the action really is a no-op.
        """
        # Special case: Previous action is None if this is the first action in the episode
        # Then we only care about criterion (1)
        if prev_action is None:
            return np.linalg.norm(action[:-1]) < threshold

        # Normal case: Check both criteria (1) and (2)
        gripper_action = action[-1]
        prev_gripper_action = prev_action[-1]
        return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

        def _parse_example(idx, ep):

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(ep["steps"]):

                task = "pick up the red block"  # hardcoded for now
                lang = self._embed([task])[0].numpy()  # embedding takes ≈0.06s

                step = jax.tree_map(lambda x: np.array(x), step)
                # pprint(jax.tree_map(lambda x: (x.shape, x.dtype), data))

                # img_shape = downscale_to_224(*image.shape[:2])
                # image = tf.image.resize(image, img_shape).numpy().astype(np.uint8)

                # needs to be reshaped because of the transforms
                img = step["observation"].pop("img")  # .astype(np.uint8)
                img = jax.tree_map(
                    lambda x: tf.image.resize(x, (224, 224)).numpy().astype(np.uint8),
                    img,
                )

                ### in the future add camera_1, camera_2, etc.
                img = {"camera_0": img['camera_0'], "wrist": img['wrist']}

                step["observation"]["image"] = img

                prop = step["observation"].pop("robot")
                prop = jax.tree_map(lambda x: x.astype(np.float32), prop)
                step["observation"]["proprio"] = prop

                spec = lambda x: jax.tree_map(lambda x: (x.shape, x.dtype), x)
                # pprint(spec(step["observation"]["image"]))

                action = step["action"].astype(np.float32)
                if self.is_noop(action):
                    continue

                episode.append(
                    {
                        "observation": step["observation"],
                        "action": step["action"].astype(np.float32),
                        "discount": 1.0,
                        "reward": float(i == (len(ep) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(ep) - 1),
                        "is_terminal": i == (len(ep) - 1),
                        "language_instruction": task,
                        "language_embedding": lang,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {}}

            # if you want to skip an example for whatever reason, simply return None
            return idx, sample

        with open(self.filtered, "r") as f:
            filtered = json.load(f)

        for path in glob.glob(paths):
            ds = tfds.builder_from_directory(path).as_dataset(split="train")

            for idx, ep in enumerate(ds):
                if not idx in filtered.get(path, {"yes": []})["yes"]:
                    continue
                yield _parse_example(f"{path}_{idx}", ep)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
