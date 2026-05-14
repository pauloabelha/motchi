"""Active sensing configuration and signal physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensingConfig:
    recharge_range: float = 12.0
    food_range: float = 10.0
    detection_threshold: float = 0.08
    base_cost: float = 0.005
    object_cost: float = 0.001


@dataclass(frozen=True)
class SensingState:
    energy_cost: float
    energy_scale: float


def quadratic_detection_strength(distance: float, sense_range: float, threshold: float, energy_scale: float = 1.0) -> float:
    if sense_range <= 0.0 or not np.isfinite(distance):
        return 0.0

    normalized_distance = float(np.clip(distance / sense_range, 0.0, 1.0))
    raw_strength = 1.0 - normalized_distance * normalized_distance
    strength = raw_strength * float(np.clip(energy_scale, 0.0, 1.0))
    return float(strength if strength >= threshold else 0.0)


def sensing_cost(config: SensingConfig, sensed_object_count: int) -> float:
    return max(0.0, config.base_cost + config.object_cost * max(0, sensed_object_count))


def sensing_scale(available_energy: float, requested_cost: float) -> float:
    if available_energy <= 0.0:
        return 0.0
    if requested_cost <= 0.0:
        return 1.0
    return float(np.clip(available_energy / requested_cost, 0.0, 1.0))
