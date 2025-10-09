from __future__ import annotations

import tensorflow_datasets as tfds

from xgym.rlds.base import XgymSingle


class XgymLiftSingle(XgymSingle):
    """DatasetBuilder for LUC XGym Single Arm"""

    VERSION = tfds.core.Version("3.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
