"""Run a Motchi ant in a visible Gymnasium Ant-v5 simulation.

Run:
    python test_ant.py
"""

from __future__ import annotations

import argparse

from motchi.body.config import AntConfig, EnvironmentConfig, RunConfig, ViewerConfig
from motchi.body.random_ant import RandomAnt
from motchi.runtime.core_drives import CoreDriveConfig
from motchi.runtime.energy import EnergyConfig
from motchi.runtime.food import FoodConfig
from motchi.runtime.sensing import SensingConfig


def build_random_ant_config(args: argparse.Namespace) -> AntConfig:
    return AntConfig(
        name="RandomAnt",
        env_id="Ant-v5",
        environment=EnvironmentConfig(
            terminate_when_unhealthy=args.terminate_when_unhealthy,
        ),
        viewer=ViewerConfig(
            width=args.width,
            height=args.height,
            camera_distance=args.camera_distance,
            camera_elevation=args.camera_elevation,
        ),
        run=RunConfig(
            seed=args.seed,
            max_steps=args.max_steps,
            reset_delay=args.reset_delay,
            telemetry_interval=max(1, args.telemetry_interval),
        ),
        energy=EnergyConfig(
            capacity=args.energy_capacity,
            base_cost=args.base_cost,
            action_cost=args.action_cost,
            recharge_rate=args.recharge_rate,
            recharge_x=args.recharge_x,
            recharge_y=args.recharge_y,
            recharge_radius=args.recharge_radius,
            empty_grace_steps=args.empty_grace_steps,
        ),
        sensing=SensingConfig(
            recharge_range=args.sensing_recharge_range,
            food_range=args.sensing_food_range,
            detection_threshold=args.sensing_threshold,
            base_cost=args.sensing_base_cost,
            object_cost=args.sensing_object_cost,
        ),
        food=FoodConfig(
            food_energy_value=args.food_energy_value,
            food_radius=args.food_radius,
            food_sense_range=args.food_sense_range,
        ),
        drives=CoreDriveConfig(
            sense_range=args.sense_range,
            gait_amplitude=args.gait_amplitude,
            wander_strength=args.wander_strength,
            seek_strength=args.seek_strength,
            steering_strength=args.steering_strength,
            frequency=args.gait_frequency,
            food_seek_strength=args.food_seek_strength,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Motchi RandomAnt with independent drives.")
    parser.add_argument("--seed", type=int, default=7, help="Initial environment seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional finite run length.")
    parser.add_argument("--reset-delay", type=float, default=0.0, help="Pause after resets.")
    parser.add_argument("--terminate-when-unhealthy", action="store_true", help="Let Ant-v5 reset when torso posture is unhealthy.")
    parser.add_argument("--energy-capacity", type=float, default=250.0, help="Maximum energy.")
    parser.add_argument("--base-cost", type=float, default=0.02, help="Energy spent every step.")
    parser.add_argument("--action-cost", type=float, default=0.20, help="Energy spent from action effort.")
    parser.add_argument("--recharge-rate", type=float, default=4.0, help="Energy restored per step in the recharge zone.")
    parser.add_argument("--recharge-x", type=float, default=3.0, help="Recharge zone X position.")
    parser.add_argument("--recharge-y", type=float, default=0.0, help="Recharge zone Y position.")
    parser.add_argument("--recharge-radius", type=float, default=2.0, help="Recharge zone radius.")
    parser.add_argument("--empty-grace-steps", type=int, default=240, help="Steps before empty energy resets the episode.")
    parser.add_argument("--telemetry-interval", type=int, default=120, help="Steps between drive status logs.")
    parser.add_argument("--sense-range", type=float, default=8.0, help="Recharge detection range.")
    parser.add_argument("--sensing-recharge-range", type=float, default=12.0, help="Recharge active sensing range.")
    parser.add_argument("--sensing-food-range", type=float, default=10.0, help="Food active sensing range.")
    parser.add_argument("--sensing-threshold", type=float, default=0.08, help="Minimum signal needed for detection.")
    parser.add_argument("--sensing-base-cost", type=float, default=0.005, help="Energy cost per sensing pass.")
    parser.add_argument("--sensing-object-cost", type=float, default=0.001, help="Additional energy cost per sensed object channel.")
    parser.add_argument("--gait-amplitude", type=float, default=0.75, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--wander-strength", type=float, default=0.35, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--seek-strength", type=float, default=0.90, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--food-seek-strength", type=float, default=1.0, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--steering-strength", type=float, default=0.45, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--gait-frequency", type=float, default=0.12, help="Reserved for future drive-consuming ants.")
    parser.add_argument("--food-energy-value", type=float, default=20.0, help="Energy restored by a small pickup.")
    parser.add_argument("--food-radius", type=float, default=0.08, help="Food touch radius.")
    parser.add_argument("--food-sense-range", type=float, default=8.0, help="Food detection range.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=900, help="Viewer window height.")
    parser.add_argument("--camera-distance", type=float, default=9.0, help="Camera distance from the ant.")
    parser.add_argument("--camera-elevation", type=float, default=-35.0, help="Camera elevation angle.")
    args = parser.parse_args()

    ant = RandomAnt(build_random_ant_config(args))
    ant.run()


if __name__ == "__main__":
    main()
