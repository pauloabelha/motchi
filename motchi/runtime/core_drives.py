"""Core drive controller for Motchi's first embodied loop."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from motchi.runtime.perception import FoodPerception, RechargePerception


@dataclass(frozen=True)
class CoreDriveConfig:
    sense_range: float = 8.0
    gait_amplitude: float = 0.75
    wander_strength: float = 0.35
    seek_strength: float = 0.90
    steering_strength: float = 0.45
    frequency: float = 0.12
    food_seek_strength: float = 1.0


@dataclass(frozen=True)
class DriveSnapshot:
    recharge: RechargePerception
    food: FoodPerception
    recharge_drive: float
    food_drive: float
    dominant_drive: str
    dominant_intensity: float
    dominant_direction_body: np.ndarray


def search_drive(perception: RechargePerception) -> float:
    """Recharge search intensity, proportional to energy depletion."""

    return float(np.clip(perception.depletion * perception.signal_strength, 0.0, 1.0))


def food_drive(perception: FoodPerception) -> float:
    """Food-seeking intensity, proportional to hunger and food signal."""

    return float(np.clip(perception.hunger_fraction * perception.signal_strength, 0.0, 1.0))


def compute_drives(recharge: RechargePerception, food: FoodPerception) -> DriveSnapshot:
    recharge_intensity = search_drive(recharge)
    food_intensity = food_drive(food)

    if food_intensity > recharge_intensity:
        return DriveSnapshot(
            recharge=recharge,
            food=food,
            recharge_drive=recharge_intensity,
            food_drive=food_intensity,
            dominant_drive="food",
            dominant_intensity=food_intensity,
            dominant_direction_body=food.direction_body,
        )

    return DriveSnapshot(
        recharge=recharge,
        food=food,
        recharge_drive=recharge_intensity,
        food_drive=food_intensity,
        dominant_drive="recharge",
        dominant_intensity=recharge_intensity,
        dominant_direction_body=recharge.direction_body,
    )


def _crawl_primitive(phase: float, steering: float, config: CoreDriveConfig) -> np.ndarray:
    """A simple hand-written Ant action pattern with left/right steering bias."""

    wave = np.sin(phase)
    counter = np.sin(phase + np.pi)
    lift = np.cos(phase)
    counter_lift = np.cos(phase + np.pi)

    left_bias = -steering * config.steering_strength
    right_bias = steering * config.steering_strength

    return config.gait_amplitude * np.array(
        [
            wave + left_bias,
            lift,
            counter + right_bias,
            counter_lift,
            counter + left_bias,
            counter_lift,
            wave + right_bias,
            lift,
        ],
        dtype=np.float64,
    )


def core_drive_action(
    env: gym.Env,
    drives: DriveSnapshot,
    step_count: int,
    config: CoreDriveConfig,
) -> np.ndarray:
    """Choose an action from core drives, without learning or policy networks."""

    if drives.dominant_drive == "food":
        active_drive = drives.food_drive
        seek_strength = config.food_seek_strength
    else:
        active_drive = drives.recharge_drive
        seek_strength = config.seek_strength

    phase = step_count * config.frequency

    steering = float(np.clip(drives.dominant_direction_body[1], -1.0, 1.0))
    crawl = _crawl_primitive(phase, steering=steering, config=config)

    random_action = env.action_space.sample().astype(np.float64)
    wander_weight = config.wander_strength * (1.0 - active_drive)
    seek_weight = seek_strength * active_drive

    action = wander_weight * random_action + seek_weight * crawl
    action += 0.10 * env.action_space.sample().astype(np.float64)
    action = np.clip(action, env.action_space.low, env.action_space.high)

    return action.astype(env.action_space.dtype)
