from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from motchi.body.config import AntConfig
from motchi.body.random_ant import RandomAnt
from motchi.runtime.actions import ActionCommand
from motchi.runtime.core_drives import compute_drives
from motchi.runtime.energy import EnergyState, spend_or_recharge
from motchi.runtime.food import FoodConfig, consume_touched_food
from motchi.runtime.perception import food_perception, recharge_perception


class _FakeData:
    def __init__(self, qpos: np.ndarray) -> None:
        self.qpos = qpos


class _FakeUnwrapped:
    def __init__(self, qpos: np.ndarray) -> None:
        self.data = _FakeData(qpos)


class _FakeEnv:
    def __init__(self, qpos: np.ndarray) -> None:
        self.unwrapped = _FakeUnwrapped(qpos)


class DriveTest(unittest.TestCase):
    def test_recharge_drive_increases_with_energy_depletion(self) -> None:
        config = AntConfig()
        qpos = np.array([2.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        high_energy = EnergyState(value=config.energy.capacity)
        low_energy = EnergyState(value=config.energy.capacity * 0.2)

        high = recharge_perception(high_energy, config.energy, config.sensing, qpos)
        low = recharge_perception(low_energy, config.energy, config.sensing, qpos)

        self.assertLess(compute_drives(high, _no_food(config, qpos)).recharge_drive, compute_drives(low, _no_food(config, qpos)).recharge_drive)

    def test_pickup_drive_increases_with_energy_depletion(self) -> None:
        config = AntConfig()
        qpos = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        food = food_perception(config.food, config.sensing, RandomAnt(config).foods, qpos)
        high_energy_recharge = recharge_perception(EnergyState.full(config.energy), config.energy, config.sensing, qpos)
        low_energy_recharge = recharge_perception(EnergyState(config.energy.capacity * 0.2), config.energy, config.sensing, qpos)

        self.assertLess(compute_drives(high_energy_recharge, food).food_drive, compute_drives(low_energy_recharge, food).food_drive)

    def test_pickup_consumption_restores_energy(self) -> None:
        ant = RandomAnt(AntConfig())
        ant.env = _FakeEnv(np.array([2.5, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        ant.energy = EnergyState(value=50.0)

        command = ActionCommand.from_motor(np.zeros(8, dtype=np.float32))
        executed = ant.motor_actuator.execute(command, ant.energy)
        with redirect_stdout(StringIO()):
            in_zone, spent = ant._update_drives_and_world(executed)

        self.assertFalse(in_zone)
        self.assertGreater(spent, 0.0)
        self.assertTrue(ant.foods[0].eaten)
        self.assertGreater(ant.energy.value, 50.0)

    def test_food_touch_helper_is_policy_free(self) -> None:
        config = FoodConfig(food_radius=0.1)
        ant = RandomAnt(AntConfig(food=config))
        consumed = consume_touched_food(np.array([2.5, 2.0], dtype=np.float64), ant.foods, config)

        self.assertEqual(consumed, [0])


def _no_food(config: AntConfig, qpos: np.ndarray):
    foods = RandomAnt(config).foods
    for food in foods:
        food.eaten = True
    return food_perception(config.food, config.sensing, foods, qpos)


if __name__ == "__main__":
    unittest.main()
