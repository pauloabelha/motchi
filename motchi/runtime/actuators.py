"""Actuator models that turn policy commands into physical energy spending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motchi.runtime.actions import ActionCommand, ExecutedAction
from motchi.runtime.energy import EnergyConfig, EnergyState


@dataclass(frozen=True)
class MotorUnitConfig:
    """Energy parameters for one low-level actuator channel."""

    name: str
    energy_multiplier: float = 1.0


@dataclass(frozen=True)
class ActuatorConfig:
    """Collection of low-level motor units for an action vector."""

    motor_units: tuple[MotorUnitConfig, ...]

    @classmethod
    def ant_v5_default(cls) -> "ActuatorConfig":
        names = (
            "front_left_hip",
            "front_left_ankle",
            "front_right_hip",
            "front_right_ankle",
            "back_left_hip",
            "back_left_ankle",
            "back_right_hip",
            "back_right_ankle",
        )
        return cls(tuple(MotorUnitConfig(name=name) for name in names))

    def multipliers_for(self, motor_size: int) -> np.ndarray:
        multipliers = np.ones(motor_size, dtype=np.float64)
        for index, unit in enumerate(self.motor_units[:motor_size]):
            multipliers[index] = max(0.0, unit.energy_multiplier)
        return multipliers


@dataclass(frozen=True)
class MotorActuator:
    """Energy-aware motor actuator for MuJoCo action vectors."""

    energy_config: EnergyConfig
    actuator_config: ActuatorConfig

    def energy_cost(self, command: ActionCommand) -> float:
        motor = np.asarray(command.motor)
        if not motor.size:
            effort = 0.0
        else:
            multipliers = self.actuator_config.multipliers_for(motor.size)
            effort = float(np.mean(np.abs(motor) * multipliers))
        return self.energy_config.base_cost + self.energy_config.action_cost * effort

    def energy_scale(self, energy: EnergyState, requested_cost: float) -> float:
        if energy.value <= 0.0:
            return 0.0
        if requested_cost <= 0.0:
            return 1.0

        affordability = energy.value / requested_cost
        reserve_fraction = energy.fraction(self.energy_config)
        scale = min(1.0, affordability, reserve_fraction * 4.0)
        return float(np.clip(scale, 0.0, 1.0))

    def execute(self, command: ActionCommand, energy: EnergyState) -> ExecutedAction:
        requested_cost = self.energy_cost(command)
        scale = self.energy_scale(energy, requested_cost)
        motor = np.asarray(command.motor) * scale
        return ExecutedAction(
            command=command,
            motor=motor,
            energy_cost=requested_cost * scale,
            energy_scale=scale,
        )
