"""Action command objects and energy-based execution gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motchi.runtime.energy import EnergyConfig, EnergyState


@dataclass(frozen=True)
class ActionCommand:
    """A proposed motor action plus the energy it would cost at full strength."""

    motor: np.ndarray
    energy_cost: float
    label: str = "motor"

    @classmethod
    def from_motor(cls, motor: np.ndarray, config: EnergyConfig, label: str = "motor") -> "ActionCommand":
        motor = np.asarray(motor)
        effort = float(np.mean(np.abs(motor))) if motor.size else 0.0
        return cls(motor=motor, energy_cost=config.base_cost + config.action_cost * effort, label=label)


@dataclass(frozen=True)
class ExecutedAction:
    """The action that actually reaches the body after energy gating."""

    command: ActionCommand
    motor: np.ndarray
    energy_cost: float
    energy_scale: float


def energy_action_scale(energy: EnergyState, config: EnergyConfig, requested_cost: float) -> float:
    """Smoothly scale action execution by available energy."""

    if energy.value <= 0.0 or requested_cost <= 0.0:
        return 0.0 if energy.value <= 0.0 else 1.0

    affordability = energy.value / requested_cost
    reserve_fraction = energy.fraction(config)
    scale = min(1.0, affordability, reserve_fraction * 4.0)
    return float(np.clip(scale, 0.0, 1.0))


def execute_action_command(command: ActionCommand, energy: EnergyState, config: EnergyConfig) -> ExecutedAction:
    scale = energy_action_scale(energy, config, command.energy_cost)
    motor = np.asarray(command.motor) * scale
    return ExecutedAction(
        command=command,
        motor=motor,
        energy_cost=command.energy_cost * scale,
        energy_scale=scale,
    )
