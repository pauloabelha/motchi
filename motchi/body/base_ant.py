"""Base Ant body with internal drives, perception, and world interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time

import gymnasium as gym
import mujoco
import numpy as np

from motchi.body.config import AntConfig
from motchi.runtime.core_drives import DriveSnapshot, compute_drives
from motchi.runtime.actions import ActionCommand, execute_action_command
from motchi.runtime.energy import EnergyState, spend_or_recharge
from motchi.runtime.food import (
    FoodItem,
    HungerState,
    consume_touched_food,
    default_food_items,
    increase_hunger,
    reduce_hunger,
)
from motchi.runtime.logging import error, info
from motchi.runtime.perception import food_perception, recharge_perception


class BaseAnt(ABC):
    """Abstract ant that owns drives but delegates action selection."""

    def __init__(self, config: AntConfig) -> None:
        self.config = config
        self.env: gym.Env | None = None
        self.energy = EnergyState.full(config.energy)
        self.hunger = HungerState()
        self.foods: list[FoodItem] = default_food_items()
        self.step_count = 0
        self.episode_count = 1
        self.was_in_recharge_zone = True

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def _env(self) -> gym.Env:
        if self.env is None:
            raise RuntimeError("Ant environment has not been created.")
        return self.env

    @abstractmethod
    def choose_action(self, drives: DriveSnapshot) -> ActionCommand:
        """Return the next proposed action command. Subclasses may ignore drives."""

    def run(self) -> None:
        self.env = self._make_env()
        self._describe_startup()
        self._reset_episode(seed=self.config.run.seed)

        try:
            while self.config.run.max_steps is None or self.step_count < self.config.run.max_steps:
                drives = self._sense_drives()
                command = self.choose_action(drives)
                executed_action = execute_action_command(command, self.energy, self.config.energy)

                observation, reward, terminated, truncated, info_dict = self._env.step(executed_action.motor)
                del observation, reward, info_dict
                self.step_count += 1

                in_recharge_zone, spent = self._update_drives_and_world(executed_action)
                self._draw_world_markers()
                self._log_telemetry(drives, in_recharge_zone, spent, executed_action.energy_scale)

                energy_failed = self.energy.empty_steps >= self.config.energy.empty_grace_steps
                if energy_failed:
                    info("Episode reset: energy depleted")

                if terminated or truncated or energy_failed:
                    self.episode_count += 1
                    info(f"Episode reset ({self.episode_count})")
                    self._reset_episode()
                    if self.config.run.reset_delay > 0:
                        time.sleep(self.config.run.reset_delay)

        except KeyboardInterrupt:
            info("Interrupted by user")
        except Exception as exc:
            error(f"{self.config.name} runtime failure after {self.step_count} steps: {exc}")
            raise
        finally:
            self._env.close()
            info("Environment closed cleanly")

    def _make_env(self) -> gym.Env:
        viewer = self.config.viewer
        return gym.make(
            self.config.env_id,
            render_mode="human",
            width=viewer.width,
            height=viewer.height,
            default_camera_config={
                "distance": viewer.camera_distance,
                "elevation": viewer.camera_elevation,
            },
        )

    def _describe_startup(self) -> None:
        env = self._env
        obs_space = env.observation_space
        action_space = env.action_space

        info(f"{self.config.name} initialized")
        info(f"Environment: {self.config.env_id}")
        info(f"Observation shape: {obs_space.shape}")
        info(f"Action shape: {action_space.shape}")
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            info(f"Action bounds low: {np.array2string(np.asarray(action_space.low), precision=3)}")
            info(f"Action bounds high: {np.array2string(np.asarray(action_space.high), precision=3)}")

        viewer = self.config.viewer
        info(f"Viewer size: {viewer.width}x{viewer.height}")
        info(f"Camera distance: {viewer.camera_distance:.1f}, elevation: {viewer.camera_elevation:.1f}")
        info(
            "Energy drive active: "
            f"capacity={self.config.energy.capacity:.1f}, "
            f"recharge=({self.config.energy.recharge_x:.1f}, {self.config.energy.recharge_y:.1f}), "
            f"radius={self.config.energy.recharge_radius:.1f}"
        )
        info(
            "Food drive active: "
            f"hunger_rate={self.config.food.hunger_rate:.3f}, "
            f"food_energy={self.config.food.food_energy_value:.1f}, "
            f"food_radius={self.config.food.food_radius:.2f}"
        )
        info("Rendering active")

    def _reset_episode(self, seed: int | None = None) -> None:
        observation, info_dict = self._env.reset(seed=seed)
        del observation, info_dict
        self.energy = EnergyState.full(self.config.energy)
        self.hunger = HungerState()
        self.foods = default_food_items()
        self.was_in_recharge_zone = True
        self._draw_world_markers()
        info("Episode reset")
        info(f"Energy: {self.energy.value:.1f}/{self.config.energy.capacity:.1f}")
        info(f"Hunger: {self.hunger.value:.1f}/{self.config.food.hunger_capacity:.1f}")

    def _sense_drives(self) -> DriveSnapshot:
        recharge = recharge_perception(
            self.energy,
            self.config.energy,
            self._env.unwrapped.data.qpos,
            sense_range=self.config.drives.sense_range,
        )
        food = food_perception(
            self.hunger,
            self.config.food,
            self.foods,
            self._env.unwrapped.data.qpos,
        )
        return compute_drives(recharge, food)

    def _torso_xy(self) -> np.ndarray:
        return np.asarray(self._env.unwrapped.data.qpos[:2], dtype=np.float64)

    def _update_drives_and_world(self, executed_action) -> tuple[bool, float]:
        self.hunger = increase_hunger(self.hunger, self.config.food)
        self.energy, in_recharge_zone, spent = spend_or_recharge(
            self.energy,
            self.config.energy,
            executed_action.motor,
            self._torso_xy(),
            action_energy_cost=executed_action.energy_cost,
        )
        consumed_food = consume_touched_food(self._torso_xy(), self.foods, self.config.food)
        for food_index in consumed_food:
            self.hunger = reduce_hunger(self.hunger, self.config.food)
            self.energy.value = float(
                np.clip(
                    self.energy.value + self.config.food.food_energy_value,
                    0.0,
                    self.config.energy.capacity,
                )
            )
            info(
                f"Food consumed ({food_index + 1}): "
                f"hunger={self.hunger.value:.1f}/{self.config.food.hunger_capacity:.1f}, "
                f"energy={self.energy.value:.1f}/{self.config.energy.capacity:.1f}"
            )

        return in_recharge_zone, spent

    def _draw_world_markers(self) -> None:
        self._add_recharge_marker()
        self._add_food_markers()

    def _viewer(self):
        renderer = self._env.unwrapped.mujoco_renderer
        if renderer.viewer is None:
            self._env.render()
        return renderer.viewer

    def _add_recharge_marker(self) -> None:
        viewer = self._viewer()
        if viewer is None:
            return

        energy = self.config.energy
        try:
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=np.array([energy.recharge_x, energy.recharge_y, 0.015]),
                size=np.array([energy.recharge_radius, energy.recharge_radius, 0.015]),
                rgba=np.array([0.1, 0.75, 0.25, 0.28]),
                label="recharge",
            )
        except Exception as exc:
            error(f"Could not draw recharge marker: {exc}")

    def _add_food_markers(self) -> None:
        viewer = self._viewer()
        if viewer is None:
            return

        for index, food in enumerate(self.foods):
            if food.eaten:
                continue
            radius = self.config.food.food_radius
            try:
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=np.array([food.xy[0], food.xy[1], radius]),
                    size=np.array([radius, radius, radius]),
                    rgba=np.array([0.05, 0.9, 0.18, 1.0]),
                    label=f"food {index + 1}",
                )
            except Exception as exc:
                error(f"Could not draw food marker: {exc}")

    def _log_telemetry(
        self,
        drives: DriveSnapshot,
        in_recharge_zone: bool,
        spent: float,
        energy_scale: float,
    ) -> None:
        zone_changed = in_recharge_zone != self.was_in_recharge_zone
        should_log = (
            self.step_count % self.config.run.telemetry_interval == 0
            or zone_changed
            or self.energy.empty
        )
        if should_log:
            info(
                f"Step {self.step_count}: energy={self.energy.value:.1f}/"
                f"{self.config.energy.capacity:.1f}, spent={spent:.3f}, "
                f"action_scale={energy_scale:.2f}, "
                f"hunger={self.hunger.value:.1f}/{self.config.food.hunger_capacity:.1f}, "
                f"in_recharge_zone={in_recharge_zone}, "
                f"recharge_drive={drives.recharge_drive:.2f}, "
                f"food_drive={drives.food_drive:.2f}, "
                f"dominant_drive={drives.dominant_drive}, "
                f"ant={self.config.name}, "
                f"recharge_distance={drives.recharge.distance:.2f}, "
                f"food_distance={drives.food.distance:.2f}"
            )
        self.was_in_recharge_zone = in_recharge_zone
