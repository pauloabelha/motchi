"""Configuration objects for Motchi ant bodies."""

from __future__ import annotations

from dataclasses import dataclass, field

from motchi.runtime.core_drives import CoreDriveConfig
from motchi.runtime.actuators import ActuatorConfig
from motchi.runtime.energy import EnergyConfig
from motchi.runtime.food import FoodConfig


@dataclass(frozen=True)
class ViewerConfig:
    width: int = 1280
    height: int = 900
    camera_distance: float = 9.0
    camera_elevation: float = -35.0


@dataclass(frozen=True)
class RunConfig:
    seed: int = 7
    max_steps: int | None = None
    reset_delay: float = 0.0
    telemetry_interval: int = 120


@dataclass(frozen=True)
class AntConfig:
    name: str = "random_ant"
    env_id: str = "Ant-v5"
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    run: RunConfig = field(default_factory=RunConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    actuators: ActuatorConfig = field(default_factory=ActuatorConfig.ant_v5_default)
    food: FoodConfig = field(default_factory=FoodConfig)
    drives: CoreDriveConfig = field(default_factory=CoreDriveConfig)
