"""RandomAnt: drive-bearing ant with a random action policy."""

from __future__ import annotations

import numpy as np

from motchi.body.base_ant import BaseAnt
from motchi.runtime.actions import ActionCommand
from motchi.runtime.core_drives import DriveSnapshot


class RandomAnt(BaseAnt):
    """An ant whose internal drives evolve but whose actions ignore them."""

    def choose_action(self, drives: DriveSnapshot) -> ActionCommand:
        del drives
        return ActionCommand.from_motor(self.action_space.sample(), label="random")
