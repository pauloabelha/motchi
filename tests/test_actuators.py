from __future__ import annotations

import unittest

import numpy as np

from motchi.body.config import AntConfig
from motchi.runtime.actions import ActionCommand
from motchi.runtime.actuators import ActuatorConfig, MotorActuator, MotorUnitConfig
from motchi.runtime.energy import EnergyState


class ActuatorTest(unittest.TestCase):
    def test_action_command_has_no_energy_cost(self) -> None:
        command = ActionCommand.from_motor(np.ones(8, dtype=np.float32), label="test")

        self.assertFalse(hasattr(command, "energy_cost"))
        self.assertEqual(command.label, "test")

    def test_motor_actuator_cost_scales_with_effort(self) -> None:
        config = AntConfig()
        actuator = MotorActuator(config.energy, config.actuators)
        rest = ActionCommand.from_motor(np.zeros(8, dtype=np.float32))
        effort = ActionCommand.from_motor(np.ones(8, dtype=np.float32))

        self.assertEqual(actuator.energy_cost(rest), config.energy.base_cost)
        self.assertGreater(actuator.energy_cost(effort), actuator.energy_cost(rest))

    def test_low_energy_smoothly_scales_action_execution(self) -> None:
        config = AntConfig()
        actuator = MotorActuator(config.energy, config.actuators)
        command = ActionCommand.from_motor(np.ones(8, dtype=np.float32))
        full = actuator.execute(command, EnergyState.full(config.energy))
        low = actuator.execute(command, EnergyState(value=config.energy.capacity * 0.1))
        empty = actuator.execute(command, EnergyState(value=0.0))

        self.assertEqual(full.energy_scale, 1.0)
        self.assertGreater(low.energy_scale, 0.0)
        self.assertLess(low.energy_scale, 1.0)
        self.assertEqual(empty.energy_scale, 0.0)

    def test_more_efficient_motor_unit_spends_less_energy(self) -> None:
        config = AntConfig()
        default = MotorActuator(config.energy, config.actuators)
        efficient_units = tuple(
            MotorUnitConfig(name=unit.name, energy_multiplier=0.25 if index == 0 else unit.energy_multiplier)
            for index, unit in enumerate(config.actuators.motor_units)
        )
        efficient = MotorActuator(config.energy, ActuatorConfig(efficient_units))
        command = ActionCommand.from_motor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertLess(efficient.energy_cost(command), default.energy_cost(command))


if __name__ == "__main__":
    unittest.main()
