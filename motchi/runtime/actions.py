"""Policy action command objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCommand:
    """A proposed motor action from a policy."""

    motor: np.ndarray
    label: str = "motor"

    @classmethod
    def from_motor(cls, motor: np.ndarray, label: str = "motor") -> "ActionCommand":
        return cls(motor=np.asarray(motor), label=label)


@dataclass(frozen=True)
class ExecutedAction:
    """The action that actually reaches the body after actuator energy gating."""

    command: ActionCommand
    motor: np.ndarray
    energy_cost: float
    energy_scale: float
