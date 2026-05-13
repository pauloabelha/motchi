from __future__ import annotations

import json
import unittest

import gymnasium as gym
import numpy as np

from motchi.body.base_ant import BaseAnt
from motchi.body.config import AntConfig
from motchi.body.random_ant import RandomAnt
from motchi.runtime.core_drives import compute_drives
from motchi.runtime.energy import EnergyState
from motchi.runtime.food import HungerState
from motchi.runtime.perception import food_perception, recharge_perception


class _FakeData:
    def __init__(self, qpos: np.ndarray) -> None:
        self.qpos = qpos


class _FakeUnwrapped:
    def __init__(self, qpos: np.ndarray) -> None:
        self.data = _FakeData(qpos)


class _FakeEnv:
    def __init__(self, qpos: np.ndarray) -> None:
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.unwrapped = _FakeUnwrapped(qpos)


class AntArchitectureTest(unittest.TestCase):
    def test_base_ant_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseAnt(AntConfig())  # type: ignore[abstract]

    def test_random_ant_inherits_base_ant(self) -> None:
        ant = RandomAnt(AntConfig())

        self.assertIsInstance(ant, BaseAnt)

    def test_random_ant_action_policy_ignores_drives(self) -> None:
        ant = RandomAnt(AntConfig())
        ant.env = _FakeEnv(np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        recharge = recharge_perception(
            ant.energy,
            ant.config.energy,
            ant.env.unwrapped.data.qpos,
            sense_range=ant.config.drives.sense_range,
        )
        food = food_perception(
            ant.hunger,
            ant.config.food,
            ant.foods,
            ant.env.unwrapped.data.qpos,
        )
        drives = compute_drives(recharge, food)

        command = ant.choose_action(drives)

        self.assertEqual(command.motor.shape, (8,))
        self.assertGreater(command.energy_cost, 0.0)
        self.assertEqual(command.label, "random")
        self.assertTrue(np.all(command.motor >= ant.action_space.low))
        self.assertTrue(np.all(command.motor <= ant.action_space.high))

    def test_random_ant_config_file_declares_inheritance(self) -> None:
        with open("configs/random_ant.json", encoding="utf-8") as file:
            config = json.load(file)

        self.assertEqual(config["ant_type"], "RandomAnt")
        self.assertEqual(config["inherits"], "BaseAnt")
        self.assertEqual(config["food"]["food_radius"], 0.08)


if __name__ == "__main__":
    unittest.main()
