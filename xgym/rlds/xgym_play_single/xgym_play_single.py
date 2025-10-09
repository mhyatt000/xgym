from __future__ import annotations

import tensorflow_datasets as tfds

from xgym.rlds.base import XgymSingle


class XgymPlaySingle(XgymSingle):
    """DatasetBuilder for LUC XGym Single Arm v2.0.0"""

    VERSION = tfds.core.Version("2.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
