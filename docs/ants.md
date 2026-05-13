# Ant Architecture

Motchi separates body drives from action policy.

## BaseAnt

`BaseAnt` is abstract. It owns the organism runtime:

- MuJoCo/Gymnasium environment lifecycle
- energy state
- hunger state
- food items
- recharge zone
- perception
- drive computation
- food consumption
- recharge handling
- reset handling
- viewer markers
- telemetry

Subclasses must implement:

```python
def choose_action(self, drives: DriveSnapshot) -> ActionCommand:
    ...
```

The `drives` snapshot is always computed before action selection. A subclass may use it, partially use it, or ignore it.

## Action Commands

Policies do not send raw motor arrays directly. They return an `ActionCommand`:

```python
ActionCommand.from_motor(motor, energy_config, label="random")
```

An action command contains:

- proposed motor output
- full-strength energy cost
- a label for telemetry/debugging

`BaseAnt` executes the command through the body energy gate. As energy gets low, motor output is smoothly scaled down. At zero energy, no motor output reaches the body.

## RandomAnt

`RandomAnt` inherits from `BaseAnt`.

It ignores its drives and samples random motor actions:

```python
class RandomAnt(BaseAnt):
    def choose_action(self, drives: DriveSnapshot) -> ActionCommand:
        del drives
        return ActionCommand.from_motor(
            self.action_space.sample(),
            self.config.energy,
            label="random",
        )
```

Even though the action policy is random, the internal body loop still updates:

- energy depletion
- recharge detection
- hunger increase
- food sensing
- food consumption
- dominant drive telemetry

That makes RandomAnt a useful baseline body: it has needs, but no intentional behavior.

## Config

The first ant config is:

```text
configs/random_ant.json
```

It declares:

```json
{
  "ant_type": "RandomAnt",
  "inherits": "BaseAnt"
}
```

The Python config types live in:

```text
motchi/body/config.py
```

The config is split into:

- `ViewerConfig`
- `RunConfig`
- `EnergyConfig`
- `FoodConfig`
- `CoreDriveConfig`
- `AntConfig`

## Adding A New Ant

Create a new subclass:

```python
from motchi.body.base_ant import BaseAnt
from motchi.runtime.core_drives import DriveSnapshot


class MyAnt(BaseAnt):
    def choose_action(self, drives: DriveSnapshot):
        # Use drives.recharge_drive, drives.food_drive, or ignore them.
        return self.action_space.sample()
```

This keeps future behavior systems, memory systems, learning policies, and procedural controllers separate from the body’s internal drives.
