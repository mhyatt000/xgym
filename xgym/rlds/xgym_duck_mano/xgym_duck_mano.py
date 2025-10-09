from __future__ import annotations

import tensorflow_datasets as tfds

from xgym.rlds.base import TFDSBaseMano


class XgymDuckMano(TFDSBaseMano):
    """DatasetBuilder for LUC XGym Mano"""

    # set VERSION and RELEASE in the parent

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        files = self._split_helper("xgym_duck_mano")
        return {"train": self._generate_examples(files)}
