"""Perception features exposed to Motchi control loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motchi.runtime.energy import EnergyConfig, EnergyState, distance_to_recharge_xy, is_in_recharge_zone
from motchi.runtime.food import FoodConfig, FoodItem, HungerState, nearest_available_food
from motchi.runtime.sensing import SensingConfig, quadratic_detection_strength


@dataclass(frozen=True)
class RechargePerception:
    energy_fraction: float
    depletion: float
    distance: float
    signal_strength: float
    direction_world: np.ndarray
    direction_body: np.ndarray
    inside_zone: bool
    detected: bool


@dataclass(frozen=True)
class FoodPerception:
    hunger_fraction: float
    distance: float
    signal_strength: float
    direction_world: np.ndarray
    direction_body: np.ndarray
    visible: bool


def yaw_from_quaternion_wxyz(quat: np.ndarray) -> float:
    """Return world yaw from a MuJoCo free-joint quaternion in w, x, y, z order."""

    w, x, y, z = np.asarray(quat, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def rotate_world_vector_to_body_xy(vector_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world-frame xy vector into the torso yaw frame."""

    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    x, y = np.asarray(vector_xy, dtype=np.float64)
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float64)


def recharge_perception(
    energy: EnergyState,
    energy_config: EnergyConfig,
    sensing_config: SensingConfig,
    torso_qpos: np.ndarray,
    sensing_energy_scale: float = 1.0,
) -> RechargePerception:
    torso_qpos = np.asarray(torso_qpos, dtype=np.float64)
    torso_xy = torso_qpos[:2]
    torso_quat = torso_qpos[3:7]

    target_xy = np.array([energy_config.recharge_x, energy_config.recharge_y], dtype=np.float64)
    delta_world = target_xy - torso_xy
    distance = distance_to_recharge_xy(torso_xy, energy_config)
    direction_world = delta_world / max(distance, 1e-6)

    yaw = yaw_from_quaternion_wxyz(torso_quat)
    direction_body = rotate_world_vector_to_body_xy(direction_world, yaw)

    energy_fraction = energy.fraction(energy_config)
    signal_strength = quadratic_detection_strength(
        distance,
        sensing_config.recharge_range,
        sensing_config.detection_threshold,
        sensing_energy_scale,
    )

    return RechargePerception(
        energy_fraction=energy_fraction,
        depletion=1.0 - energy_fraction,
        distance=distance,
        signal_strength=signal_strength,
        direction_world=direction_world,
        direction_body=direction_body,
        inside_zone=is_in_recharge_zone(torso_xy, energy_config),
        detected=signal_strength > 0.0,
    )


def food_perception(
    hunger: HungerState,
    food_config: FoodConfig,
    sensing_config: SensingConfig,
    foods: list[FoodItem],
    torso_qpos: np.ndarray,
    sensing_energy_scale: float = 1.0,
) -> FoodPerception:
    torso_qpos = np.asarray(torso_qpos, dtype=np.float64)
    torso_xy = torso_qpos[:2]
    torso_quat = torso_qpos[3:7]

    food_index, distance = nearest_available_food(torso_xy, foods)
    hunger_fraction = hunger.fraction(food_config)

    if food_index is None:
        return FoodPerception(
            hunger_fraction=hunger_fraction,
            distance=float("inf"),
            signal_strength=0.0,
            direction_world=np.zeros(2, dtype=np.float64),
            direction_body=np.zeros(2, dtype=np.float64),
            visible=False,
        )

    delta_world = foods[food_index].xy - torso_xy
    direction_world = delta_world / max(distance, 1e-6)
    yaw = yaw_from_quaternion_wxyz(torso_quat)
    direction_body = rotate_world_vector_to_body_xy(direction_world, yaw)
    signal_strength = quadratic_detection_strength(
        distance,
        min(food_config.food_sense_range, sensing_config.food_range),
        sensing_config.detection_threshold,
        sensing_energy_scale,
    )

    return FoodPerception(
        hunger_fraction=hunger_fraction,
        distance=distance,
        signal_strength=signal_strength,
        direction_world=direction_world,
        direction_body=direction_body,
        visible=signal_strength > 0.0,
    )
