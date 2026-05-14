"""Energy pickup state for Motchi worlds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoodConfig:
    food_energy_value: float = 20.0
    food_radius: float = 0.08
    food_sense_range: float = 8.0


@dataclass
class FoodItem:
    xy: np.ndarray
    eaten: bool = False


def default_food_items() -> list[FoodItem]:
    return [
        FoodItem(np.array([2.5, 2.0], dtype=np.float64)),
        FoodItem(np.array([-2.5, 2.0], dtype=np.float64)),
        FoodItem(np.array([3.5, -1.5], dtype=np.float64)),
        FoodItem(np.array([-3.0, -2.0], dtype=np.float64)),
    ]


def nearest_available_food(torso_xy: np.ndarray, foods: list[FoodItem]) -> tuple[int | None, float]:
    best_index: int | None = None
    best_distance = float("inf")

    for index, food in enumerate(foods):
        if food.eaten:
            continue
        distance = float(np.linalg.norm(food.xy - torso_xy))
        if distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index, best_distance


def consume_touched_food(
    torso_xy: np.ndarray,
    foods: list[FoodItem],
    config: FoodConfig,
) -> list[int]:
    consumed: list[int] = []

    for index, food in enumerate(foods):
        if food.eaten:
            continue
        distance = float(np.linalg.norm(food.xy - torso_xy))
        if distance <= config.food_radius:
            food.eaten = True
            consumed.append(index)

    return consumed
