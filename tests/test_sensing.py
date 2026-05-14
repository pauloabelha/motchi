from __future__ import annotations

import unittest

import numpy as np

from motchi.body.config import AntConfig
from motchi.body.random_ant import RandomAnt
from motchi.runtime.energy import EnergyState
from motchi.runtime.perception import food_perception, recharge_perception
from motchi.runtime.sensing import quadratic_detection_strength, sensing_cost, sensing_scale


class SensingTest(unittest.TestCase):
    def test_quadratic_detection_falls_with_distance(self) -> None:
        near = quadratic_detection_strength(distance=2.0, sense_range=10.0, threshold=0.0)
        far = quadratic_detection_strength(distance=8.0, sense_range=10.0, threshold=0.0)

        self.assertGreater(near, far)
        self.assertAlmostEqual(near, 0.96)
        self.assertAlmostEqual(far, 0.36)

    def test_detection_threshold_hides_weak_signal(self) -> None:
        signal = quadratic_detection_strength(distance=9.9, sense_range=10.0, threshold=0.08)

        self.assertEqual(signal, 0.0)

    def test_recharge_perception_can_be_undetected_but_still_inside_zone(self) -> None:
        config = AntConfig()
        qpos = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        perception = recharge_perception(
            EnergyState.full(config.energy),
            config.energy,
            config.sensing,
            qpos,
            sensing_energy_scale=0.0,
        )

        self.assertTrue(perception.inside_zone)
        self.assertFalse(perception.detected)
        self.assertEqual(perception.signal_strength, 0.0)

    def test_food_perception_uses_thresholded_sensing(self) -> None:
        config = AntConfig()
        qpos = np.array([100.0, 100.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        perception = food_perception(RandomAnt(config).hunger, config.food, config.sensing, RandomAnt(config).foods, qpos)

        self.assertFalse(perception.visible)
        self.assertEqual(perception.signal_strength, 0.0)

    def test_sensing_cost_and_scale_depend_on_energy(self) -> None:
        config = AntConfig()
        cost = sensing_cost(config.sensing, sensed_object_count=5)

        self.assertGreater(cost, config.sensing.base_cost)
        self.assertEqual(sensing_scale(available_energy=cost * 2.0, requested_cost=cost), 1.0)
        self.assertEqual(sensing_scale(available_energy=0.0, requested_cost=cost), 0.0)

    def test_live_sensing_spends_energy(self) -> None:
        ant = RandomAnt(AntConfig())
        ant.env = type(
            "FakeEnv",
            (),
            {
                "unwrapped": type(
                    "FakeUnwrapped",
                    (),
                    {"data": type("FakeData", (), {"qpos": np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])})()},
                )()
            },
        )()
        before = ant.energy.value

        ant._sense_drives(spend_energy=True)

        self.assertLess(ant.energy.value, before)
        self.assertGreater(ant.last_sensing.energy_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
