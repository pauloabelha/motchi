from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from motchi.body.config import AntConfig
from motchi.body.random_ant import RandomAnt
from motchi.runtime.actions import ActionCommand, execute_action_command
from motchi.runtime.core_drives import compute_drives
from motchi.runtime.energy import EnergyState, spend_or_recharge
from motchi.runtime.food import FoodConfig, HungerState, consume_touched_food, reduce_hunger
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

        high = recharge_perception(high_energy, config.energy, qpos, sense_range=config.drives.sense_range)
        low = recharge_perception(low_energy, config.energy, qpos, sense_range=config.drives.sense_range)

        self.assertLess(compute_drives(high, _no_food(config, qpos)).recharge_drive, compute_drives(low, _no_food(config, qpos)).recharge_drive)

    def test_food_drive_increases_with_hunger(self) -> None:
        config = AntConfig()
        qpos = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        low_hunger = HungerState(value=0.0)
        high_hunger = HungerState(value=config.food.hunger_capacity)

        low = food_perception(low_hunger, config.food, RandomAnt(config).foods, qpos)
        high = food_perception(high_hunger, config.food, RandomAnt(config).foods, qpos)

        recharge = recharge_perception(EnergyState.full(config.energy), config.energy, qpos, sense_range=config.drives.sense_range)

        self.assertLess(compute_drives(recharge, low).food_drive, compute_drives(recharge, high).food_drive)

    def test_food_consumption_reduces_hunger_and_restores_energy(self) -> None:
        ant = RandomAnt(AntConfig())
        ant.env = _FakeEnv(np.array([2.5, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        ant.energy = EnergyState(value=50.0)
        ant.hunger = HungerState(value=80.0)

        command = ActionCommand.from_motor(np.zeros(8, dtype=np.float32), ant.config.energy)
        executed = execute_action_command(command, ant.energy, ant.config.energy)
        with redirect_stdout(StringIO()):
            in_zone, spent = ant._update_drives_and_world(executed)

        self.assertFalse(in_zone)
        self.assertGreater(spent, 0.0)
        self.assertTrue(ant.foods[0].eaten)
        self.assertLess(ant.hunger.value, 80.0)
        self.assertGreater(ant.energy.value, 50.0)

    def test_food_touch_and_hunger_helpers_are_policy_free(self) -> None:
        config = FoodConfig(food_radius=0.1, food_hunger_value=10.0)
        ant = RandomAnt(AntConfig(food=config))
        consumed = consume_touched_food(np.array([2.5, 2.0], dtype=np.float64), ant.foods, config)
        hunger = reduce_hunger(HungerState(value=50.0), config)

        self.assertEqual(consumed, [0])
        self.assertEqual(hunger.value, 40.0)

    def test_action_command_cost_scales_with_effort(self) -> None:
        config = AntConfig()
        rest = ActionCommand.from_motor(np.zeros(8, dtype=np.float32), config.energy)
        effort = ActionCommand.from_motor(np.ones(8, dtype=np.float32), config.energy)

        self.assertEqual(rest.energy_cost, config.energy.base_cost)
        self.assertGreater(effort.energy_cost, rest.energy_cost)

    def test_low_energy_smoothly_scales_action_execution(self) -> None:
        config = AntConfig()
        command = ActionCommand.from_motor(np.ones(8, dtype=np.float32), config.energy)
        full = execute_action_command(command, EnergyState.full(config.energy), config.energy)
        low = execute_action_command(command, EnergyState(value=config.energy.capacity * 0.1), config.energy)
        empty = execute_action_command(command, EnergyState(value=0.0), config.energy)

        self.assertEqual(full.energy_scale, 1.0)
        self.assertGreater(low.energy_scale, 0.0)
        self.assertLess(low.energy_scale, 1.0)
        self.assertEqual(empty.energy_scale, 0.0)


def _no_food(config: AntConfig, qpos: np.ndarray):
    foods = RandomAnt(config).foods
    for food in foods:
        food.eaten = True
    return food_perception(HungerState(), config.food, foods, qpos)


if __name__ == "__main__":
    unittest.main()
