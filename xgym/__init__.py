from __future__ import annotations

from pathlib import Path

from xgym import calibrate, utils, viz
from xgym.calibrate import urdf
from xgym.calibrate.urdf import robot
from xgym.names import *  # noqa
from xgym.utils import logger

try:
    from xgym import nodes
except ImportError as ex:
    logger.error(ex)
    logger.error("Cannot import xgym.nodes")

logger.info("Setting up xgym")
