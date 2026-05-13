"""Energy bookkeeping for embodied simulation harnesses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyConfig:
    capacity: float = 250.0
    base_cost: float = 0.02
    action_cost: float = 0.20
    recharge_rate: float = 4.0
    recharge_x: float = 0.0
    recharge_y: float = 0.0
    recharge_radius: float = 1.5
    empty_grace_steps: int = 240


@dataclass
class EnergyState:
    value: float
    empty_steps: int = 0

    @classmethod
    def full(cls, config: EnergyConfig) -> "EnergyState":
        return cls(value=config.capacity)

    @property
    def empty(self) -> bool:
        return self.value <= 0.0

    def fraction(self, config: EnergyConfig) -> float:
        return float(np.clip(self.value / config.capacity, 0.0, 1.0))


def distance_to_recharge_xy(xy: np.ndarray, config: EnergyConfig) -> float:
    center = np.array([config.recharge_x, config.recharge_y], dtype=np.float64)
    return float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - center))


def is_in_recharge_zone(xy: np.ndarray, config: EnergyConfig) -> bool:
    return distance_to_recharge_xy(xy, config) <= config.recharge_radius


def spend_or_recharge(
    state: EnergyState,
    config: EnergyConfig,
    action: np.ndarray,
    torso_xy: np.ndarray,
    action_energy_cost: float | None = None,
) -> tuple[EnergyState, bool, float]:
    """Advance energy by one simulation step.

    Returns the updated state, whether the body is in the recharge zone, and
    the energy spent this step before any recharge is applied.
    """

    in_zone = is_in_recharge_zone(torso_xy, config)
    if action_energy_cost is None:
        effort = float(np.mean(np.abs(action))) if action.size else 0.0
        spent = config.base_cost + config.action_cost * effort
    else:
        spent = max(0.0, float(action_energy_cost))

    next_value = state.value - spent
    if in_zone:
        next_value += config.recharge_rate

    next_value = float(np.clip(next_value, 0.0, config.capacity))
    empty_steps = state.empty_steps + 1 if next_value <= 0.0 else 0

    return EnergyState(value=next_value, empty_steps=empty_steps), in_zone, spent
