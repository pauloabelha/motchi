# Motchi MuJoCo Foundation

Motchi is a minimal, reproducible MuJoCo experimentation stack for future embodied AI and synthetic organism work. This phase only validates simulation, rendering, random control, resets, and a clean project structure.

## WSL Ubuntu System Dependencies

Run these once in WSL Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  libgl1 libgl1-mesa-dri libglfw3 libglew2.2 \
  libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
  mesa-utils
```

For WSLg, a GUI session should expose `DISPLAY` automatically. You can check it with:

```bash
echo "$DISPLAY"
glxinfo -B
```

## Python Setup

From this directory:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For exact reproduction of this resolved environment, use:

```bash
python -m pip install -r requirements-lock.txt
```

## Validate MuJoCo

Headless physics smoke test:

```bash
source venv/bin/activate
python validate_mujoco.py --headless
```

Visible GLFW viewer smoke test:

```bash
source venv/bin/activate
python validate_mujoco.py
```

## Run Ant-v5

```bash
source venv/bin/activate
python test_ant.py
```

Expected startup logs include:

```text
[INFO] Ant-v5 initialized
[INFO] Observation shape: (105,)
[INFO] Action shape: (8,)
[INFO] Rendering active
[INFO] Episode reset
```

Gymnasium Ant-v5 currently reports a `(105,)` observation vector by default. The action vector is `(8,)`, one actuator command per ant motor.

## Energy And Recharge

`test_ant.py` now includes a simple organism-style energy loop:

- each action spends energy
- a green circular recharge zone is drawn on the floor
- the ant recharges while its torso is inside that zone
- if energy reaches zero, motor commands are gated to zero
- if energy remains empty for too long, the episode resets
- the default `core` controller increases recharge seeking in proportion to energy depletion

Run with defaults:

```bash
source venv/bin/activate
python test_ant.py
```

`test_ant.py` currently instantiates `RandomAnt`, which inherits from `BaseAnt`.

- `BaseAnt` owns the body runtime, energy drive, food drive, perception, markers, consumption, recharge, reset, and telemetry.
- `RandomAnt` only supplies the action policy: random motor commands.
- This means drives exist even though RandomAnt does not intentionally use them.

Relevant files:

```text
motchi/body/config.py
motchi/body/base_ant.py
motchi/body/random_ant.py
configs/random_ant.json
docs/ants.md
```

The default viewer is `1280x900` with a zoomed-out camera. You can adjust it:

```bash
python test_ant.py --width 1600 --height 1000 --camera-distance 12 --camera-elevation -40
```

Run a short, obvious depletion test:

```bash
python test_ant.py \
  --max-steps 400 \
  --energy-capacity 20 \
  --action-cost 1.5 \
  --base-cost 0.1 \
  --recharge-rate 0.2 \
  --recharge-radius 0.75 \
  --empty-grace-steps 20
```

Move the recharge zone:

```bash
python test_ant.py --recharge-x 2.0 --recharge-y 0.0 --recharge-radius 1.0
```

RandomAnt always uses random motor actions. Motchi still updates drives:

- energy is spent or recharged
- hunger rises
- food is consumed if touched
- recharge and food drive intensities are computed
- telemetry reports the dominant drive
- action commands are smoothly scaled by available energy

The ant class decides whether drives influence motor actions. `RandomAnt` ignores them. Future ants can inherit from `BaseAnt` and use the same drives differently.

Run a random-action drive demo where the recharge region is away from the ant:

```bash
python test_ant.py \
  --recharge-x 3 \
  --recharge-y 0 \
  --energy-capacity 30 \
  --action-cost 0.7 \
  --base-cost 0.05 \
  --recharge-rate 2 \
  --sense-range 8
```

The drive layer exposes intentionally simple internal signals:

```text
depletion = 1 - energy_fraction
search_drive = depletion * recharge_signal_strength
hunger_drive = hunger_fraction * food_signal_strength
```

For `RandomAnt`, these signals are observed and logged but do not bias movement. Food and recharge can still be fulfilled randomly if the ant happens to reach them. This is not reinforcement learning; it is the first separation between body drives and action policy.

Action policies return `ActionCommand` objects rather than raw arrays. A command contains proposed motor output, but not energy cost. Energy cost belongs to the body actuator: `BaseAnt` owns a `MotorActuator`, and the actuator computes spending from the motor command. Its low-level motor units each have an `energy_multiplier`, so a more efficient leg or joint spends less energy for the same command. Low energy produces weak executed actions, and zero energy produces no executed action.

Food appears as small green spheres, about a tenth of the ant's body scale. When the ant torso touches one:

- the food is marked eaten and no longer renders
- hunger is reduced
- energy is restored

Run a food-heavy demo:

```bash
python test_ant.py \
  --hunger-rate 1.0 \
  --food-radius 1.0 \
  --food-hunger-value 50 \
  --food-energy-value 30
```

Run a quick verification where food is intentionally easy to touch:

```bash
python test_ant.py \
  --max-steps 220 \
  --food-radius 3.0 \
  --hunger-rate 1.0 \
  --food-hunger-value 50 \
  --food-energy-value 30
```

## Project Layout

```text
motchi/
├── venv/
├── experiments/
├── logs/
├── videos/
├── configs/
├── motchi/
│   ├── body/
│   ├── runtime/
│   └── utils/
├── validate_mujoco.py
├── requirements.txt
└── test_ant.py
```

This layout leaves room for later replay logging, memory systems, drives, procedural behavior, learning systems, and robotics experiments without introducing heavy abstractions early.

## Tests

Run the non-GUI test suite:

```bash
source venv/bin/activate
python -m unittest discover -s tests
```

The tests cover:

- `BaseAnt` abstraction
- `RandomAnt` inheritance
- random action shape/bounds
- config file inheritance declaration
- recharge drive increasing with energy depletion
- food drive increasing with hunger
- food consumption reducing hunger and restoring energy
- policy commands staying independent from actuator energy cost
- actuator-computed energy costs
- per-motor actuator efficiency
- smooth low-energy actuator scaling
